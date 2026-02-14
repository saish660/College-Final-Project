package com.smartclassroom.smartclassroom;

public class StudentRegistrationRequest {
    private String name;
    private String roll_no;
    private String email;
    private String password;

    public StudentRegistrationRequest(String name, String roll_no, String email, String password) {
        this.name = name;
        this.roll_no = roll_no;
        this.email = email;
        this.password = password;
    }

    // Getters and setters
}
