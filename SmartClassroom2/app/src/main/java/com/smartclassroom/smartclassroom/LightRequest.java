package com.smartclassroom.smartclassroom;

import com.google.gson.annotations.SerializedName;

import java.util.List;

public class LightRequest {
    @SerializedName("status")
    private final boolean status;
    @SerializedName("arguments")
    private final List<String> arguments;

    public LightRequest(boolean status, List<String> arguments) {
        this.status = status;
        this.arguments = arguments;
    }

    public LightRequest(boolean status) {
        this.status = status;
        this.arguments = null;
    }
}
