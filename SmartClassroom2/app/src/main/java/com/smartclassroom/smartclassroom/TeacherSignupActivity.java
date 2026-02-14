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

public class TeacherSignupActivity extends AppCompatActivity {

    private EditText editTextName, editTextEmail, editTextPassword;
    private Button buttonSignup;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_teacher_signup);

        editTextName = findViewById(R.id.editTextTeacherName);
        editTextEmail = findViewById(R.id.editTextTeacherEmail);
        editTextPassword = findViewById(R.id.editTextTeacherPassword);
        buttonSignup = findViewById(R.id.buttonTeacherSignup);

        buttonSignup.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                signup();
            }
        });
    }

    private void signup() {
        String name = editTextName.getText().toString().trim();
        String email = editTextEmail.getText().toString().trim();
        String password = editTextPassword.getText().toString().trim();

        if (name.isEmpty() || email.isEmpty() || password.isEmpty()) {
            Toast.makeText(this, "Please fill all fields", Toast.LENGTH_SHORT).show();
            return;
        }

        TeacherRegistrationRequest request = new TeacherRegistrationRequest(name, email, password);
        Call<RegistrationResponse> call = ApiClient.getApiService().createTeacher(request);

        call.enqueue(new Callback<RegistrationResponse>() {
            @Override
            public void onResponse(Call<RegistrationResponse> call, Response<RegistrationResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    Toast.makeText(TeacherSignupActivity.this, response.body().getMessage(), Toast.LENGTH_SHORT).show();
                    if (!response.body().isError()) {
                        // Navigate to login or dashboard
                        Intent intent = new Intent(TeacherSignupActivity.this, TeacherLoginActivity.class);
                        startActivity(intent);
                        finish();
                    }
                } else {
                    Toast.makeText(TeacherSignupActivity.this, "Signup failed", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<RegistrationResponse> call, Throwable t) {
                Toast.makeText(TeacherSignupActivity.this, "An error occurred", Toast.LENGTH_SHORT).show();
            }
        });
    }
}
