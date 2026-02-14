
package com.smartclassroom.smartclassroom;

import android.content.Context;
import android.content.SharedPreferences;

public class SessionManager {

    private static final String PREF_NAME = "SmartClassroomPref";
    private static final String KEY_IS_LOGGED_IN = "isLoggedIn";
    private static final String KEY_USER_TYPE = "userType";
    private static final String KEY_ROLL_NO = "rollNo";

    private SharedPreferences pref;
    private SharedPreferences.Editor editor;
    private Context _context;

    public SessionManager(Context context) {
        this._context = context;
        pref = _context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
        editor = pref.edit();
    }

    public void setLogin(boolean isLoggedIn, String userType, String rollNo) {
        editor.putBoolean(KEY_IS_LOGGED_IN, isLoggedIn);
        editor.putString(KEY_USER_TYPE, userType);
        editor.putString(KEY_ROLL_NO, rollNo);
        editor.apply();
    }

    public boolean isLoggedIn() {
        return pref.getBoolean(KEY_IS_LOGGED_IN, false);
    }

    public String getUserType() {
        return pref.getString(KEY_USER_TYPE, "");
    }

    public String getRollNo() {
        return pref.getString(KEY_ROLL_NO, null);
    }

    public void logoutUser() {
        editor.clear();
        editor.apply();
    }
}
