package com.smartclassroom.smartclassroom;

public class RegistrationResponse {
    private String message;
    private String error;

    public String getMessage() {
        return message;
    }

    public String getError() {
        return error;
    }

    public boolean isError() {
        return error != null;
    }
}
