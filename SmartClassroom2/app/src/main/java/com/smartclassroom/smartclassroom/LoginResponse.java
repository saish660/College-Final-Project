package com.smartclassroom.smartclassroom;

import com.google.gson.annotations.SerializedName;

public class LoginResponse {
    private boolean error;
    private String message;
    private String token;
    @SerializedName("roll_no")
    private String rollNo;

    public boolean isError() {
        return error;
    }

    public String getMessage() {
        return message;
    }

    public String getToken() {
        return token;
    }

    public String getRollNo() {
        return rollNo;
    }
}
