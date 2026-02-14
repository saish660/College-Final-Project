package com.smartclassroom.smartclassroom;

import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import java.util.ArrayList;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class StudentDashboardActivity extends AppCompatActivity {

    private RecyclerView recyclerViewAttendance;
    private AttendanceAdapter adapter;
    private List<AttendanceRecord> attendanceList;
    private SessionManager sessionManager;
    private TextView textViewWelcome;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_student_dashboard);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        sessionManager = new SessionManager(getApplicationContext());

        textViewWelcome = findViewById(R.id.textViewWelcome);
        recyclerViewAttendance = findViewById(R.id.recyclerViewAttendance);
        recyclerViewAttendance.setLayoutManager(new LinearLayoutManager(this));

        attendanceList = new ArrayList<>();
        adapter = new AttendanceAdapter(attendanceList, false);
        recyclerViewAttendance.setAdapter(adapter);

        findViewById(R.id.buttonLogout).setOnClickListener(v -> {
            sessionManager.logoutUser();
            Intent intent = new Intent(StudentDashboardActivity.this, MainActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
            startActivity(intent);
            finish();
        });

        FloatingActionButton refreshButton = findViewById(R.id.buttonRefresh);
        refreshButton.setOnClickListener(v -> fetchAttendance());

        fetchAttendance();
    }

    private void fetchAttendance() {
        String rollNo = sessionManager.getRollNo();
        if (rollNo == null) {
            Toast.makeText(this, "Roll number not found", Toast.LENGTH_SHORT).show();
            return;
        }

        ApiService apiService = ApiClient.getApiService();
        Call<List<AttendanceRecord>> call = apiService.getAttendance(rollNo);

        call.enqueue(new Callback<List<AttendanceRecord>>() {
            @Override
            public void onResponse(Call<List<AttendanceRecord>> call, Response<List<AttendanceRecord>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    attendanceList.clear();
                    attendanceList.addAll(response.body());
                    adapter.notifyDataSetChanged();

                    if (!attendanceList.isEmpty()) {
                        AttendanceRecord firstRecord = attendanceList.get(0);
                        if (firstRecord.getName() != null && !firstRecord.getName().isEmpty()) {
                            textViewWelcome.setText("Welcome, " + firstRecord.getName() + "!");
                        }
                    }
                } else {
                    Toast.makeText(StudentDashboardActivity.this, "Failed to fetch attendance", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<List<AttendanceRecord>> call, Throwable t) {
                Toast.makeText(StudentDashboardActivity.this, "An error occurred", Toast.LENGTH_SHORT).show();
            }
        });
    }
}
