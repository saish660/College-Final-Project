package com.smartclassroom.smartclassroom;

import com.google.gson.annotations.SerializedName;

public class AttendanceRecord {

    @SerializedName("student_id")
    private Integer studentId;

    @SerializedName("roll_no")
    private String rollNo;

    @SerializedName("date")
    private String date;

    @SerializedName("status")
    private String status;

    @SerializedName("captured_at")
    private String capturedAt;

    @SerializedName("teacher_email")
    private String teacherEmail;

    @SerializedName("confidence")
    private float confidence;

    @SerializedName("name")
    private String name;

    @SerializedName("samples")
    private int samples;

    public Integer getStudentId() {
        return studentId;
    }

    public String getRollNo() {
        return rollNo;
    }

    public String getDate() {
        return date;
    }

    public String getStatus() {
        return status;
    }

    public String getCapturedAt() {
        return capturedAt;
    }

    public String getTeacherEmail() {
        return teacherEmail;
    }

    public float getConfidence() {
        return confidence;
    }

    public String getName() {
        return name;
    }

    public int getSamples() {
        return samples;
    }
}
