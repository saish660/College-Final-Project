package com.smartclassroom.smartclassroom;

import android.content.Context;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.List;
import java.util.Locale;

public class AttendanceListAdapter extends ArrayAdapter<AttendanceRecord> {

    public AttendanceListAdapter(@NonNull Context context, @NonNull List<AttendanceRecord> records) {
        super(context, 0, records);
    }

    @NonNull
    @Override
    public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
        Log.d("AttendanceListAdapter", "getView called for position: " + position + ". Total count: " + getCount());

        if (convertView == null) {
            convertView = LayoutInflater.from(getContext()).inflate(R.layout.list_item_attendance, parent, false);
        }

        AttendanceRecord record = getItem(position);

        TextView textViewName = convertView.findViewById(R.id.textViewName);
        TextView textViewRollNo = convertView.findViewById(R.id.textViewRollNo);
        TextView textViewConfidence = convertView.findViewById(R.id.textViewConfidence);

        if (record != null) {
            if (record.getName() != null) {
                textViewName.setText(record.getName());
            } else {
                textViewName.setText("Unregistered");
            }
            
            if (record.getRollNo() != null) {
                textViewRollNo.setText(record.getRollNo());
            } else {
                textViewRollNo.setText("Roll number not available");
            }

            textViewConfidence.setText(String.format(Locale.getDefault(), "Confidence: %.2f%%", record.getConfidence() * 100));

            // Safe logging
            if (record.getRollNo() != null) {
                Log.d("AttendanceListAdapter", "Processing record for roll number: " + record.getRollNo());
            } else {
                Log.w("AttendanceListAdapter", "Record has a null roll number at position: " + position);
            }
        } else {
            Log.e("AttendanceListAdapter", "AttendanceRecord is null at position: " + position);
        }

        return convertView;
    }
}
