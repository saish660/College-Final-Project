package com.smartclassroom.smartclassroom;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private Button buttonTeacher, buttonStudent;
    private SessionManager sessionManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        sessionManager = new SessionManager(getApplicationContext());

        if (sessionManager.isLoggedIn()) {
            String userType = sessionManager.getUserType();
            if (userType.equals("teacher")) {
                startActivity(new Intent(MainActivity.this, DashboardActivity.class));
            } else if (userType.equals("student")) {
                startActivity(new Intent(MainActivity.this, StudentDashboardActivity.class));
            }
            finish();
        }

        buttonTeacher = findViewById(R.id.buttonTeacher);
        buttonStudent = findViewById(R.id.buttonStudent);

        buttonTeacher.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Assuming you have a TeacherLoginActivity
                startActivity(new Intent(MainActivity.this, TeacherLoginActivity.class));
            }
        });

        buttonStudent.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this, StudentLoginActivity.class));
            }
        });
    }
}
