package com.smartclassroom.smartclassroom;

public class AttendanceRequest {
    private String teacher_email;
    private String status;

    public AttendanceRequest(String teacher_email, String status) {
        this.teacher_email = teacher_email;
        this.status = status;
    }

    public String getTeacher_email() {
        return teacher_email;
    }

    public void setTeacher_email(String teacher_email) {
        this.teacher_email = teacher_email;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }
}
