package com.smartclassroom.smartclassroom;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class StudentLoginActivity extends AppCompatActivity {

    private EditText editTextEmail, editTextPassword;
    private Button buttonLogin;
    private TextView textViewSignup;
    private SessionManager sessionManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_student_login);

        sessionManager = new SessionManager(getApplicationContext());

        editTextEmail = findViewById(R.id.editTextStudentEmail);
        editTextPassword = findViewById(R.id.editTextStudentPassword);
        buttonLogin = findViewById(R.id.buttonStudentLogin);
        textViewSignup = findViewById(R.id.textViewStudentSignup);

        buttonLogin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                login();
            }
        });

        textViewSignup.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(StudentLoginActivity.this, StudentSignupActivity.class));
            }
        });
    }

    private void login() {
        String email = editTextEmail.getText().toString().trim();
        String password = editTextPassword.getText().toString().trim();

        if (email.isEmpty() || password.isEmpty()) {
            Toast.makeText(this, "Please fill all fields", Toast.LENGTH_SHORT).show();
            return;
        }

        LoginRequest request = new LoginRequest(email, password);
        Call<LoginResponse> call = ApiClient.getApiService().studentLogin(request);

        call.enqueue(new Callback<LoginResponse>() {
            @Override
            public void onResponse(Call<LoginResponse> call, Response<LoginResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    if (!response.body().isError()) {
                        Log.d("StudentLoginActivity", "Roll No from response: " + response.body().getRollNo());
                        sessionManager.setLogin(true, "student", response.body().getRollNo());
                        Toast.makeText(StudentLoginActivity.this, "Login successful", Toast.LENGTH_SHORT).show();
                        Intent intent = new Intent(StudentLoginActivity.this, StudentDashboardActivity.class);
                        startActivity(intent);
                        finish();
                    } else {
                        Toast.makeText(StudentLoginActivity.this, response.body().getMessage(), Toast.LENGTH_SHORT).show();
                    }
                } else {
                    Toast.makeText(StudentLoginActivity.this, "Login failed", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<LoginResponse> call, Throwable t) {
                Toast.makeText(StudentLoginActivity.this, "An error occurred", Toast.LENGTH_SHORT).show();
            }
        });
    }
}
