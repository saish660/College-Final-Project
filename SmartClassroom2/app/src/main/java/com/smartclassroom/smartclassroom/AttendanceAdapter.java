package com.smartclassroom.smartclassroom;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.List;
import java.util.Locale;

public class AttendanceAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {

    private final List<AttendanceRecord> attendanceList;
    private final boolean isTeacherView;

    private static final int VIEW_TYPE_TEACHER = 1;
    private static final int VIEW_TYPE_STUDENT = 2;

    private final DateTimeFormatter inputFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss[.SSSSSS]");
    private final DateTimeFormatter outputFormatter = DateTimeFormatter.ofPattern("MMM d, yyyy, h:mm a", Locale.getDefault());


    public AttendanceAdapter(List<AttendanceRecord> attendanceList, boolean isTeacherView) {
        this.attendanceList = attendanceList;
        this.isTeacherView = isTeacherView;
    }

    @Override
    public int getItemViewType(int position) {
        return isTeacherView ? VIEW_TYPE_TEACHER : VIEW_TYPE_STUDENT;
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        if (viewType == VIEW_TYPE_TEACHER) {
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_attendance_teacher, parent, false);
            return new TeacherViewHolder(view);
        } else {
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_attendance, parent, false);
            return new StudentViewHolder(view);
        }
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        AttendanceRecord record = attendanceList.get(position);
        if (holder.getItemViewType() == VIEW_TYPE_TEACHER) {
            ((TeacherViewHolder) holder).bind(record);
        } else {
            ((StudentViewHolder) holder).bind(record, inputFormatter, outputFormatter);
        }
    }

    @Override
    public int getItemCount() {
        return attendanceList.size();
    }

    public static class StudentViewHolder extends RecyclerView.ViewHolder {
        TextView textViewDate;
        TextView textViewStatus;
        TextView textViewConfidence;

        public StudentViewHolder(@NonNull View itemView) {
            super(itemView);
            textViewDate = itemView.findViewById(R.id.textViewDate);
            textViewStatus = itemView.findViewById(R.id.textViewStatus);
            textViewConfidence = itemView.findViewById(R.id.textViewConfidence);
        }

        public void bind(AttendanceRecord record, DateTimeFormatter inputFormatter, DateTimeFormatter outputFormatter) {
            if (record == null) {
                textViewDate.setText("Invalid record");
                textViewStatus.setText("N/A");
                textViewConfidence.setText("N/A");
                return;
            }

            String capturedAt = record.getCapturedAt();
            if (capturedAt != null) {
                try {
                    LocalDateTime dateTime = LocalDateTime.parse(capturedAt, inputFormatter);
                    textViewDate.setText(dateTime.format(outputFormatter));
                } catch (DateTimeParseException e) {
                    textViewDate.setText(capturedAt);
                }
            } else {
                textViewDate.setText("N/A");
            }

            String status = record.getStatus();
            if (status != null) {
                textViewStatus.setText(status);
            } else {
                textViewStatus.setText("N/A");
            }
            textViewConfidence.setText("");
        }
    }

    public static class TeacherViewHolder extends RecyclerView.ViewHolder {
        TextView textViewName;
        TextView textViewRollNo;
        TextView textViewConfidence;

        public TeacherViewHolder(@NonNull View itemView) {
            super(itemView);
            textViewName = itemView.findViewById(R.id.textViewName);
            textViewRollNo = itemView.findViewById(R.id.textViewRollNo);
            textViewConfidence = itemView.findViewById(R.id.textViewConfidence);
        }

        public void bind(AttendanceRecord record) {
            if (record == null) {
                textViewName.setText("Invalid record");
                textViewRollNo.setText("");
                textViewConfidence.setText("");
                return;
            }

            if (record.getName() != null) {
                textViewName.setText(record.getName());
                textViewRollNo.setText(record.getRollNo());
            } else {
                textViewName.setText("Unregistered");
                if (record.getRollNo() != null) {
                    textViewRollNo.setText(record.getRollNo());
                } else {
                    textViewRollNo.setText("");
                }
            }
            textViewConfidence.setText(String.format(Locale.getDefault(), "Confidence: %.2f%%", record.getConfidence() * 100));
        }
    }
}
