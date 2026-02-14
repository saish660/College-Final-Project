package com.smartclassroom.smartclassroom;

import android.os.Bundle;
import android.util.Log;
import android.widget.ListView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class AttendanceActivity extends AppCompatActivity {

    private ListView attendanceListView;
    private AttendanceListAdapter attendanceAdapter;
    private List<AttendanceRecord> studentRecords = new ArrayList<>();

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        setContentView(R.layout.activity_attendance);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        attendanceListView = findViewById(R.id.attendanceListView);
        attendanceAdapter = new AttendanceListAdapter(this, studentRecords);
        attendanceListView.setAdapter(attendanceAdapter);

        FloatingActionButton cameraBtn = findViewById(R.id.cameraBtn);
        cameraBtn.setOnClickListener(v -> {
            ApiService apiService = ApiClient.getApiService();
            String teacherEmail = getSharedPreferences("AUTH", MODE_PRIVATE).getString("email", null);

            if (teacherEmail == null) {
                Toast.makeText(AttendanceActivity.this, "Error: Not logged in. Please log in again.", Toast.LENGTH_LONG).show();
                return;
            }

            AttendanceRequest request = new AttendanceRequest(teacherEmail, "present");

            apiService.startAttendance(request).enqueue(new Callback<AttendanceResponse>() {
                @Override
                public void onResponse(@NonNull Call<AttendanceResponse> call, @NonNull Response<AttendanceResponse> response) {
                    if (response.isSuccessful() && response.body() != null) {
                        AttendanceResponse attendanceResponse = response.body();
                        studentRecords.clear();

                        if (attendanceResponse.getRecords() != null) {
                            studentRecords.addAll(attendanceResponse.getRecords());
                        }

                        attendanceAdapter.notifyDataSetChanged();
                        String message = attendanceResponse.getMessage();
                        if (message == null) {
                            message = "Attendance process completed.";
                        }
                        Toast.makeText(AttendanceActivity.this, message, Toast.LENGTH_SHORT).show();
                    } else {
                        String errorBody = "";
                        try {
                            if (response.errorBody() != null) {
                                errorBody = response.errorBody().string();
                            }
                        } catch (IOException e) {
                            Log.e("AttendanceActivity", "Error reading error body", e);
                        }
                        String errorMsg = "Failed to start attendance: " + response.code() + " " + errorBody;
                        Toast.makeText(AttendanceActivity.this, errorMsg, Toast.LENGTH_LONG).show();
                        Log.d("AttendanceActivity", errorMsg);
                    }
                }

                @Override
                public void onFailure(@NonNull Call<AttendanceResponse> call, @NonNull Throwable t) {
                    Log.e("AttendanceActivity", "Error starting attendance", t);
                    Toast.makeText(AttendanceActivity.this, "Error: " + t.getMessage(), Toast.LENGTH_SHORT).show();
                }
            });
        });
    }
}
