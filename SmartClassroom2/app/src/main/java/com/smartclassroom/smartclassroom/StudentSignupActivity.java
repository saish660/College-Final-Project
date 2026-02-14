package com.smartclassroom.smartclassroom;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class StudentSignupActivity extends AppCompatActivity {

    private EditText editTextName, editTextRollNo, editTextEmail, editTextPassword;
    private Button buttonSignup;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_student_signup);

        editTextName = findViewById(R.id.editTextStudentName);
        editTextRollNo = findViewById(R.id.editTextStudentRollNo);
        editTextEmail = findViewById(R.id.editTextStudentEmail);
        editTextPassword = findViewById(R.id.editTextStudentPassword);
        buttonSignup = findViewById(R.id.buttonStudentSignup);

        buttonSignup.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                signup();
            }
        });
    }

    private void signup() {
        String name = editTextName.getText().toString().trim();
        String rollNo = editTextRollNo.getText().toString().trim();
        String email = editTextEmail.getText().toString().trim();
        String password = editTextPassword.getText().toString().trim();

        if (name.isEmpty() || rollNo.isEmpty() || email.isEmpty() || password.isEmpty()) {
            Toast.makeText(this, "Please fill all fields", Toast.LENGTH_SHORT).show();
            return;
        }

        StudentRegistrationRequest request = new StudentRegistrationRequest(name, rollNo, email, password);
        Call<RegistrationResponse> call = ApiClient.getApiService().createStudent(request);

        call.enqueue(new Callback<RegistrationResponse>() {
            @Override
            public void onResponse(Call<RegistrationResponse> call, Response<RegistrationResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    Toast.makeText(StudentSignupActivity.this, response.body().getMessage(), Toast.LENGTH_SHORT).show();
                    if (!response.body().isError()) {
                        // Navigate to login or dashboard
                        Intent intent = new Intent(StudentSignupActivity.this, StudentLoginActivity.class);
                        startActivity(intent);
                        finish();
                    }
                } else {
                    Toast.makeText(StudentSignupActivity.this, "Signup failed", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<RegistrationResponse> call, Throwable t) {
                Toast.makeText(StudentSignupActivity.this, "An error occurred", Toast.LENGTH_SHORT).show();
            }
        });
    }
}
