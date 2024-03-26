using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEditor.UI;
using UnityEditor.UIElements;
using System;

public class SicknessManager : MonoBehaviour
{
    public int predicted_sickness;
    public bool prevent_sickness;

    public Renderer vignette;
    public GameObject MoveProvider;
    public GameObject TurnProvider;
    public GameObject MajorSicknessNotif;


    private ActionBasedSnapTurnProvider SnapTurnProvider;
    private ActionBasedContinuousTurnProvider ContinuousTurnProvider;
    private MaterialPropertyBlock vignette_settings;
    [SerializeField]
    private float time_very_sick = 0f;

    // Start is called before the first frame update
    void Start()
    {
        SnapTurnProvider = TurnProvider.GetComponent<ActionBasedSnapTurnProvider>();
        ContinuousTurnProvider = TurnProvider.GetComponent<ActionBasedContinuousTurnProvider>();
        vignette_settings = new MaterialPropertyBlock();
    }

    // Update is called once per frame
    void Update()
    {
        if (predicted_sickness == 4) time_very_sick += Time.deltaTime; // Increment/Decrement time_very_sick.
        else if (predicted_sickness < 3) time_very_sick = 0f;
        else time_very_sick -= Time.deltaTime;

        if (time_very_sick > 30f)  // User has been very sick for 30s.
        {
            MajorSicknessNotif.SetActive(true);
        }
        else MajorSicknessNotif.SetActive(false);

        time_very_sick = Math.Clamp(time_very_sick, 0f, 35f);  // Clamp time_very_sick (cant have been very sick for less than 0s, 35 so 5s till notification dissapears).
    }

    public void ToggleSnapTurn(bool isUsingSnapTurn)
    {
        SnapTurnProvider.enabled = isUsingSnapTurn;
        ContinuousTurnProvider.enabled = !isUsingSnapTurn;
    }

    public void SetVignette(float vignette_size)
    {
        vignette_settings.SetFloat("_ApertureSize", vignette_size);

        vignette.SetPropertyBlock(vignette_settings);
    }

    public void ChangeSickness(int recieved_sickness)
    {
        predicted_sickness = recieved_sickness;

        SetVignette(1f - ((1f / 6f) * (float)predicted_sickness)); // 0 sickness is 1 vignette size, 4 sickness is 0.5 vignette size.

        if (predicted_sickness > 2) ToggleSnapTurn(true);
        else ToggleSnapTurn(false);
    }

    public void ChangeSickness(float recieved_sickness)
    {
        ChangeSickness((int)recieved_sickness);
    }
}
