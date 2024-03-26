using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class VRDebug : MonoBehaviour
{
    public SicknessPredictor sp;

    public TMP_Text CamDebug;
    public TMP_Text CtrlDebug;
    public TMP_Text PoseDebug;
    public RawImage ImageDebug;


    //Start is called before the first frame update
    void Start()
    {

    }

    //Update is called once per frame
    void Update()
    {

        CamDebug.text = $"Camera Data\n  Projection\n{PrintMatrix(sp.camdata.GetRange(0, 16), 4)}\n  View\n{PrintMatrix(sp.camdata.GetRange(16, 16), 4)}";
        CtrlDebug.text = $"ControllerData\n  Joystick\n    Left: ({sp.ctrldata[0]}, {sp.ctrldata[1]})\n    Right: ({sp.ctrldata[2]}, {sp.ctrldata[3]})\n  Touchpad\n    Left: ({sp.ctrldata[4]}, {sp.ctrldata[5]})\n    Right: ({sp.ctrldata[6]}, {sp.ctrldata[7]})\n  Trigger\n    Left: {sp.ctrldata[8]}\n    Right: {sp.ctrldata[9]}";
        PoseDebug.text = $"PoseData\n  Hmd\n    deviceToAbsoluteTracking\n{PrintMatrix(sp.posedata.GetRange(0, 12), 6)}\n    Velocity: ({sp.posedata[12]}, {sp.posedata[13]}, {sp.posedata[14]})\n    AngularVelocity: ({sp.posedata[15]}, {sp.posedata[16]}, {sp.posedata[17]})\n  Left Controller\n    deviceToAbsoluteTracking\n{PrintMatrix(sp.posedata.GetRange(18, 12), 6)}\n    Velocity: ({sp.posedata[30]}, {sp.posedata[31]}, {sp.posedata[32]})\n    AngularVelocity: ({sp.posedata[33]}, {sp.posedata[34]}, {sp.posedata[35]})\n  Right Controller\n    deviceToAbsoluteTracking\n{PrintMatrix(sp.posedata.GetRange(36, 12), 6)}\n    Velocity: ({sp.posedata[48]}, {sp.posedata[49]}, {sp.posedata[50]})\n    AngularVelocity: ({sp.posedata[51]}, {sp.posedata[52]}, {sp.posedata[53]})";
        ImageDebug.texture = sp.headsetImage;
    }

    string PrintMatrix(List<float> values, int indent)
    {
        string sin = "";
        for (int i = 0; i < indent; i++) { sin += " "; }
        string s = "";
        for (int i = 0; i < values.Count; i++)
        {
            if (i % 4 == 3)
                s += $"{sin}{values[i]}\n";
            else
                s += $"{sin}{values[i]}";
        }
        return s;
    }
}
