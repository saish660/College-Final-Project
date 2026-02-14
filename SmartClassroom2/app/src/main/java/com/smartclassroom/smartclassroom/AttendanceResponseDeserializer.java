package com.smartclassroom.smartclassroom;

import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.Collections;
import java.util.List;

public class AttendanceResponseDeserializer implements JsonDeserializer<AttendanceResponse> {
    @Override
    public AttendanceResponse deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) throws JsonParseException {
        JsonObject jsonObject = json.getAsJsonObject();
        Gson gson = new Gson();

        AttendanceResponse response = new AttendanceResponse();

        if (jsonObject.has("message")) {
            response.setMessage(jsonObject.get("message").getAsString());
        }
        if (jsonObject.has("saved")) {
            response.setSaved(jsonObject.get("saved").getAsInt());
        }
        if (jsonObject.has("missing_students")) {
            response.setMissingStudents(jsonObject.get("missing_students").getAsInt());
        }

        JsonElement recordsElement = jsonObject.get("records");
        if (recordsElement != null && !recordsElement.isJsonNull()) {
            if (recordsElement.isJsonArray()) {
                List<AttendanceRecord> records = gson.fromJson(recordsElement, new TypeToken<List<AttendanceRecord>>(){}.getType());
                response.setRecords(records);
            } else if (recordsElement.isJsonObject()) {
                AttendanceRecord singleRecord = gson.fromJson(recordsElement, AttendanceRecord.class);
                response.setRecords(Collections.singletonList(singleRecord));
            }
        }

        return response;
    }
}
