package com.smartclassroom.smartclassroom;

import com.google.gson.JsonObject;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;

public interface ApiService {
    @POST("login")
    Call<LoginResponse> login(@Body LoginRequest request);

    @POST("lights/toggle")
    Call<LightResponse> toggleLights(@Body LightRequest request);

    @POST("attendance/process")
    Call<AttendanceResponse> startAttendance(@Body AttendanceRequest request);

    @POST("create-teacher")
    Call<RegistrationResponse> createTeacher(@Body TeacherRegistrationRequest request);

    @POST("create-student")
    Call<RegistrationResponse> createStudent(@Body StudentRegistrationRequest request);

    @POST("student/login")
    Call<LoginResponse> studentLogin(@Body LoginRequest request);

    @GET("attendance/roll/{roll_no}")
    Call<List<AttendanceRecord>> getAttendance(@Path("roll_no") String rollNo);
}
