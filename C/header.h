// ============ Login Module ============
int login(char* user);  // Login. Returns 1 if successful, and stores the username

// ============ Date Selection ============
int choose_date();  // Returns 0 (today), 1 (tomorrow), 2 (day after tomorrow), or -1 (exit)
void get_date(int offset, char* date_str);  // Get the date string of the day with the given offset

// ============ Timeslot Selection ============
int choose_slot(int day_offset);  // Returns a timeslot index (0~7), or -1 to exit

// ============ Classroom Matching ============
int find_room(int size, int date_offset, int slot_index, char* room);  // Find the smallest suitable classroom ID

// ============ Booking Writing ============
int save_booking(char* room, int date_offset, int slot_index, char* user);