package com.smartclassroom.smartclassroom;

import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class DashboardActivity extends AppCompatActivity {

    private SessionManager sessionManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dashboard);

        sessionManager = new SessionManager(getApplicationContext());

        findViewById(R.id.markAttendanceCameraBtn).setOnClickListener(v ->
                startActivity(new Intent(this, AttendanceActivity.class)));

        findViewById(R.id.controlLightsBtn).setOnClickListener(v -> showLightControlDialog());

        findViewById(R.id.logoutBtn).setOnClickListener(v -> {
            sessionManager.logoutUser();
            Intent intent = new Intent(DashboardActivity.this, MainActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
            startActivity(intent);
            finish();
        });
    }

    private void showLightControlDialog() {
        final CharSequence[] options = {"ON", "OFF"};
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Control Lights");
        builder.setItems(options, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int item) {
                if (options[item].equals("ON")) {
                    controlLights(true);
                } else if (options[item].equals("OFF")) {
                    controlLights(false);
                }
            }
        });
        builder.show();
    }

    private void controlLights(boolean status) {
        ApiService api = ApiClient.getApiService();
        LightRequest request = new LightRequest(status);

        api.toggleLights(request).enqueue(new Callback<LightResponse>() {
            @Override
            public void onResponse(Call<LightResponse> call, Response<LightResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    LightResponse lightResponse = response.body();
                    String state = lightResponse.isLightsOn() ? "ON" : "OFF";
                    Toast.makeText(DashboardActivity.this, "Lights are now " + state + ". Process: " + lightResponse.getProcessState(), Toast.LENGTH_LONG).show();
                } else {
                    Toast.makeText(DashboardActivity.this, "Failed to control lights", Toast.LENGTH_SHORT).show();
                    Log.e("LIGHTS", "Failed to control lights: " + response.code() + " " + response.message());
                }
            }

            @Override
            public void onFailure(Call<LightResponse> call, Throwable t) {
                Toast.makeText(DashboardActivity.this, "Network error", Toast.LENGTH_SHORT).show();
                Log.e("LIGHTS", "Network failure", t);
            }
        });
    }
}
