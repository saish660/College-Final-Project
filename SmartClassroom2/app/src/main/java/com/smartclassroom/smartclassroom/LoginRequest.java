package com.smartclassroom.smartclassroom;

public class LoginRequest {
    public String email;
    public String password;

    public LoginRequest() {}  // ← ADD THIS

    public LoginRequest(String email, String password) {
        this.email = email;
        this.password = password;
    }
}
