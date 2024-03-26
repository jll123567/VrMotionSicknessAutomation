using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.XR.Interaction.Toolkit;

public class CoasterManager : MonoBehaviour
{
    public ParentConstraint rig_contstraint;
    public SicknessManager sickness_manager;
    private Animator anim;
    public int seated = 0;
    public bool seating = false;


    // Start is called before the first frame update
    void Start()
    {
        anim = this.GetComponent<Animator>();
    }

    // Update is called once per frame
    void Update()
    {
        AnimatorClipInfo[] actual_clip = anim.GetCurrentAnimatorClipInfo(anim.GetLayerIndex("ActualCar"));
        AnimatorClipInfo[] circle_clip = anim.GetCurrentAnimatorClipInfo(anim.GetLayerIndex("CircleCar"));

        if (actual_clip.Length == 0) // No clip playing, stopped.
        {
            if (seated == 1 && !seating) ExitCoaster(0);
        }
        else anim.ResetTrigger("ActualStart");
        
        if (circle_clip.Length == 0)
        {
            if (seated == 2 && !seating) ExitCoaster(1);
        }
        else anim.ResetTrigger("CircleStart");

        if(actual_clip.Length != 0 || circle_clip.Length !=0 ) seating = false;
    }

    public void EnterCoaster(int which)
    {
        if (seated != 0) return;
        seated = which+1;
        seating = true;
        Debug.Log("Entering Coaster");

        rig_contstraint.constraintActive = true;

        ConstraintSource constraint_source = rig_contstraint.GetSource(which);
        constraint_source.weight = 1f;
        rig_contstraint.SetSource(which, constraint_source);


        rig_contstraint.weight = 1f;
        if(which == 0) anim.SetTrigger("ActualStart");
        else anim.SetTrigger("CircleStart");
    }

    public void ExitCoaster(int which)
    {
        if (seated == 0) return;
        seated = 0;

        Debug.Log("Exiting Coaster");

        ConstraintSource actual_source = rig_contstraint.GetSource(which);
        actual_source.weight = 0f;
        rig_contstraint.SetSource(which, actual_source);

        rig_contstraint.weight = 0f;
        rig_contstraint.constraintActive = false;
    }

}
