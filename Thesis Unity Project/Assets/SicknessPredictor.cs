using System.Collections;
using System.Collections.Generic;
using System.Text;
using System;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR;
using Unity.Jobs;
using Unity.Collections;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;

public class ModelInput
{
    [VectorType(100 * 116)]
    [ColumnName("input_1")]
    public float[] Numeric { get; set; }

    [VectorType(100 * 131 * 256 * 3)]
    [ColumnName("input_2")]
    public float[] Image { get; set; }
}

public class ModelOutput
{
    [VectorType(5)]
    [ColumnName("dense_5")]
    public float[] Data { get; set; }
}

public class SicknessPredictor : MonoBehaviour
{

    public Camera main_camera;

    public InputActionProperty[] joysticks;
    public InputActionProperty[] touchpads;
    public InputActionProperty[] triggers;

    public UnityEngine.XR.InputDevice hmd;
    public UnityEngine.XR.InputDevice l_hand;
    public UnityEngine.XR.InputDevice r_hand;

    public RenderTexture LeftEye;
    public RenderTexture RightEye;

    public List<float> camdata;
    public List<float> ctrldata;
    public List<float> posedata;
    public Texture2D headsetImage;

    private NativeArray<float> last100Numeric;
    private NativeArray<float> last100Image;

    public string modelPath = "";
    

    public int predictionThisFrame = 0;

    bool jobRan = false;
    int bufferCount = 0;
    PredictJob predictJobInstance;
    JobHandle predictJobHandle;




    // Start is called before the first frame update
    void Start()
    {
        hmd = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(XRNode.Head);
        l_hand = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        r_hand = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(XRNode.RightHand);

        foreach (InputActionProperty joy in joysticks)
        {
            joy.action.Enable();
        }
        foreach (InputActionProperty touch in touchpads)
        {
            touch.action.Enable();
        }
        foreach (InputActionProperty trigger in triggers)
        {
            trigger.action.Enable();
        }
        last100Numeric = new NativeArray<float>(100*116, Allocator.Persistent);
        last100Image = new NativeArray<float>(100 * 131 * 256 * 3, Allocator.Persistent);
    }

    // Update is called once per frame
    void Update()
    {
        if (!hmd.isValid) hmd = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(XRNode.Head);  // If input devices arent ready, ready them.

        if (!l_hand.isValid) l_hand = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);

        if (!r_hand.isValid) r_hand = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(XRNode.RightHand);

        if (predictJobHandle.IsCompleted)
        {
            camdata = GetCameraData();
            ctrldata = GetControllerData();
            posedata = GetPoseData();
            headsetImage = GetHeadsetImage();

            List<float> numData = new List<float>(camdata);
            numData.AddRange(ctrldata);
            numData.AddRange(posedata);

                
            // Handle completed job.
            predictJobHandle.Complete();
            predictionThisFrame = predictJobInstance.prediciton;
            if (jobRan)
            {
                last100Numeric.CopyFrom(predictJobInstance.last100Numeric);
                last100Image.CopyFrom(predictJobInstance.last100Image);
                predictJobInstance.newImage.Dispose();
            }
            bufferCount = (int)Math.Min(bufferCount+1, 100);
            predictJobInstance = new PredictJob()
            {
                last100Numeric = this.last100Numeric,
                last100Image = this.last100Image,
                newNumeric = new NativeArray<float>(numData.ToArray(), Allocator.TempJob),
                newImage = headsetImage.GetPixelData<ushort>(0),
                modelPath = new NativeArray<byte>(Encoding.ASCII.GetBytes(this.modelPath),Allocator.TempJob),
                bufferCount = this.bufferCount,
                prediciton = -1
            };
            Debug.Log("Job Schedule");
            predictJobHandle = predictJobInstance.Schedule();
            jobRan = true;
        }

    }

    void OnDestroy()
    {
        predictJobHandle.Complete();
        last100Image.Dispose();
        last100Numeric.Dispose();
    }

    public float[] FlattenNumeric(List<List<float>> last100Data)
    {
        List<float> flatlist = new List<float>();

        foreach (List<float> l in last100Data)
        {
            foreach (float f in l)
            {
                flatlist.Add(f);
            }
        }

        return flatlist.ToArray();
    }

    List<float> GetCameraData()
    {
        Matrix4x4 projection = main_camera.GetStereoProjectionMatrix(Camera.StereoscopicEye.Left);
        Matrix4x4 view = main_camera.GetStereoViewMatrix(Camera.StereoscopicEye.Left);
        List<float> mout = new List<float>();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                mout.Add(projection[i, j]);
            }
        }
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                mout.Add(view[i, j]);
            }
        }
        return mout;
    }

    List<float> GetControllerData()
    {

        Vector2 joy_l = joysticks[0].action.ReadValue<Vector2>();
        Vector2 joy_r = joysticks[1].action.ReadValue<Vector2>();
        Vector2 touch_l = touchpads[0].action.ReadValue<Vector2>();
        Vector2 touch_r = touchpads[1].action.ReadValue<Vector2>();
        float trigger_l = triggers[0].action.ReadValue<float>();
        float trigger_r = triggers[1].action.ReadValue<float>();


        return new List<float>{
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,  // Headset, 0 for all axis.
            0f, 0f, joy_l.x, 0f, joy_l.y, 0f, 0f, 0f, 0f, 0f, // Left Controller, only joystick x,y
            0f, 0f, joy_r.x, 0f, joy_r.y, 0f, 0f, 0f, 0f, 0f // Right Controller, only joystick x,y
        };
    }

    List<float> PositionRotationToMatrix(Vector3 pos, Quaternion rot)
    {
        List<float> flatmat;
        if (rot.w == 0f && rot.x == 0f && rot.y == 0f && rot.z == 0f) flatmat = new List<float>(new float[12]); // Fill with 0 for an invalid quaterion.
        else
        {
            Matrix4x4 mat = Matrix4x4.TRS(pos, rot, new Vector3(1f, 1f, 1f));
            flatmat = new List<float>();
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    flatmat.Add(mat[i, j]);
                }
            }
        }
        return flatmat;
    }

    List<float> GetPoseData()
    {
        // hmd: DevicePosition:3, DeviceRotation:4, DeviceVelocity:5, DeviceAngularVelocity:6
        // left/right: DevicePostion:20, DeviceRotation:21, DeviceVelocity: 22, DeviceAngularVelocity:23
        List<float> mout = new List<float>();
        Vector3 hmd_pos = new Vector3();
        Quaternion hmd_rot = new Quaternion();
        Vector3 hmd_vel = new Vector3();
        Vector3 hmd_avl = new Vector3();

        if (hmd.isValid)
        {
            List<InputFeatureUsage> hmd_usages = new List<InputFeatureUsage>();
            hmd.TryGetFeatureUsages(hmd_usages);

            hmd.TryGetFeatureValue(hmd_usages[3].As<Vector3>(), out hmd_pos);
            hmd.TryGetFeatureValue(hmd_usages[4].As<Quaternion>(), out hmd_rot);
            hmd.TryGetFeatureValue(hmd_usages[5].As<Vector3>(), out hmd_vel);
            hmd.TryGetFeatureValue(hmd_usages[6].As<Vector3>(), out hmd_avl);
        }

        Vector3 l_pos = new Vector3();
        Quaternion l_rot = new Quaternion();
        Vector3 l_vel = new Vector3();
        Vector3 l_avl = new Vector3();

        if (l_hand.isValid)
        {
            List<InputFeatureUsage> l_usages = new List<InputFeatureUsage>();
            l_hand.TryGetFeatureUsages(l_usages);

            l_hand.TryGetFeatureValue(l_usages[20].As<Vector3>(), out l_pos);
            l_hand.TryGetFeatureValue(l_usages[21].As<Quaternion>(), out l_rot);
            l_hand.TryGetFeatureValue(l_usages[22].As<Vector3>(), out l_vel);
            l_hand.TryGetFeatureValue(l_usages[23].As<Vector3>(), out l_avl);
        }

        Vector3 r_pos = new Vector3();
        Quaternion r_rot = new Quaternion();
        Vector3 r_vel = new Vector3();
        Vector3 r_avl = new Vector3();

        if (r_hand.isValid)
        {
            List<InputFeatureUsage> r_usages = new List<InputFeatureUsage>();
            r_hand.TryGetFeatureUsages(r_usages);

            r_hand.TryGetFeatureValue(r_usages[20].As<Vector3>(), out r_pos);
            r_hand.TryGetFeatureValue(r_usages[21].As<Quaternion>(), out r_rot);
            r_hand.TryGetFeatureValue(r_usages[22].As<Vector3>(), out r_vel);
            r_hand.TryGetFeatureValue(r_usages[23].As<Vector3>(), out r_avl);
        }

        mout.AddRange(PositionRotationToMatrix(hmd_pos, hmd_rot));
        mout.Add(hmd_vel.x);
        mout.Add(hmd_vel.y);
        mout.Add(hmd_vel.z);
        mout.Add(hmd_avl.x);
        mout.Add(hmd_avl.y);
        mout.Add(hmd_avl.z);

        mout.AddRange(PositionRotationToMatrix(l_pos, l_rot));
        mout.Add(l_vel.x);
        mout.Add(l_vel.y);
        mout.Add(l_vel.z);
        mout.Add(l_avl.x);
        mout.Add(l_avl.y);
        mout.Add(l_avl.z);

        mout.AddRange(PositionRotationToMatrix(r_pos, r_rot));
        mout.Add(r_vel.x);
        mout.Add(r_vel.y);
        mout.Add(r_vel.z);
        mout.Add(r_avl.x);
        mout.Add(r_avl.y);
        mout.Add(r_avl.z);

        return mout;
    }

    public Texture2D GetHeadsetImage()
    {
        Texture2D fullImage = new Texture2D(256, 131, TextureFormat.RGB48, false);

        RenderTexture.active = LeftEye;
        fullImage.ReadPixels(new Rect(0, 0, LeftEye.width, LeftEye.height), 0, 0, false);

        RenderTexture.active = RightEye;
        fullImage.ReadPixels(new Rect(0, 0, RightEye.width, RightEye.height), 128, 0, false);

        fullImage.Apply();
        return fullImage;

    }

    struct PredictJob: IJob {
        public NativeArray<float> last100Numeric;
        [DeallocateOnJobCompletion]
        public NativeArray<float> newNumeric;
        public NativeArray<float> last100Image;
        public NativeArray<ushort> newImage;


        public int bufferCount;
        public int prediciton;
        [DeallocateOnJobCompletion]
        public NativeArray<byte> modelPath;

        public void Execute()
        {
            NativeArray<float>.Copy(last100Numeric, 116, last100Numeric, 0, 99*116);  // Remove oldest(first) 116 values
            NativeArray<float>.Copy(newNumeric, 0, last100Numeric, (99*116), 116);  // Add newest 116 values to end

            NativeArray<float>.Copy(last100Image, 131 * 256 * 3, last100Image, 0, 99 * 131 * 256 * 3); // Remove oldest(first) image.
            for(int i=0; i<newImage.Length; i++)
            {
                last100Image[i + (99 * 131 * 256 * 3)] = ((float)newImage[i]) / 65535f;
            }

            

            // Predict.
            if (bufferCount >= 100)
            {  // Exit early if not enough observations.
                ModelInput mi = new ModelInput();
                mi.Numeric = last100Numeric.ToArray();
                mi.Image = last100Image.ToArray();

                string modelPathS = Encoding.ASCII.GetString(modelPath.ToArray());
                string[] outputColumnNames = new[] { "dense_5" };
                string[] inputColumnNames = new[] { "input_1", "input_2" };
                var mlContext = new MLContext();
                var pipeline = mlContext.Transforms.ApplyOnnxModel(outputColumnNames, inputColumnNames, modelPathS);
                var dataView = mlContext.Data.LoadFromEnumerable<ModelInput>(new[] { mi });
                //var transformedValues = pipeline.Fit(dataView).Transform(dataView);
                //var output = mlContext.Data.CreateEnumerable<ModelOutput>(transformedValues, reuseRowObject: false);

                prediciton = 0; // Dummy value, pls remove.
                Debug.Log("Job End");

                //Single[] outArr = outThisFrame.Data;

                //int greatestPrediction = 0;
                //for(int i = 1; i<5; i++)
                //{
                //    if(outThisFrame)
                //}
            }
            return;
        }

    }

    public float[] GetHeadsetImagePixels(Texture2D image)
    {
        NativeArray<ushort> img_array = image.GetPixelData<ushort>(0);
        float[] img_list = new float[img_array.Length];

        for(int i = 0; i < img_array.Length; i++){
            img_list[i] = ((float)img_array[i]) / 65535f;
        }

        return img_list;
    }
}
