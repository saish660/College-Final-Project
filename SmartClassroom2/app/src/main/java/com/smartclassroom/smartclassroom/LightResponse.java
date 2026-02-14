package com.smartclassroom.smartclassroom;

import com.google.gson.annotations.SerializedName;

public class LightResponse {
    @SerializedName("lights_on")
    private final boolean lightsOn;
    @SerializedName("process_state")
    private final String processState;

    public LightResponse(boolean lightsOn, String processState) {
        this.lightsOn = lightsOn;
        this.processState = processState;
    }

    public boolean isLightsOn() {
        return lightsOn;
    }

    public String getProcessState() {
        return processState;
    }
}
