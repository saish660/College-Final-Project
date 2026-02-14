package com.smartclassroom.smartclassroom;

import com.google.gson.annotations.SerializedName;

import java.util.List;

public class AttendanceResponse {
    @SerializedName("message")
    private String message;

    @SerializedName("saved")
    private int saved;

    @SerializedName("missing_students")
    private int missingStudents;

    @SerializedName("records")
    private List<AttendanceRecord> records;

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public int getSaved() {
        return saved;
    }

    public void setSaved(int saved) {
        this.saved = saved;
    }

    public int getMissingStudents() {
        return missingStudents;
    }

    public void setMissingStudents(int missingStudents) {
        this.missingStudents = missingStudents;
    }

    public List<AttendanceRecord> getRecords() {
        return records;
    }

    public void setRecords(List<AttendanceRecord> records) {
        this.records = records;
    }
}
