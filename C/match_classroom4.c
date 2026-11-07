#include <stdio.h>
#include <string.h>
#include "header.h"

// Configuration constants
#define MAX_ROOMS 20
#define MAX_NAME 20
#define DAYS 3        // Today + next two days
#define SLOTS 8       // Time slots per day
#define NO_ROOM_FOUND -1  // Sentinel value for no match

// Represents a classroom with its schedule and bookings
typedef struct {
    char id[10];         // Classroom identifier
    int capacity;        // Max number of people
    int booked[DAYS][SLOTS];           // 0 = available, 1 = booked
    char user[DAYS][SLOTS][MAX_NAME];  // Who booked each slot
} Classroom;

// Converts date string to day offset (0=today, 1=tomorrow, 2=day after)
int date_to_offset(const char* date_str) {
    for (int offset = 0; offset < DAYS; offset++) {
        char temp[20];
        get_date(offset, temp);  // Get date string for this offset
        if (strcmp(temp, date_str) == 0) {
            return offset;  // Found matching date
        }
    }
    return -1;  // Date not within range
}

// Loads booking data from classroom-specific file
void load_classroom_status(Classroom* room) {
    char filename[20];
    sprintf(filename, "%s.txt", room->id);  // Build filename from room ID

    FILE* file = fopen(filename, "r");
    if (!file) return;  // No booking file exists yet

    char date[20], username[20];
    int slot, offset;

    // Process each booking record
    while (fscanf(file, "%s %d %s", date, &slot, username) == 3) {
        offset = date_to_offset(date);
        // Only process valid slots and offsets
        if (offset >= 0 && offset < DAYS && slot >= 0 && slot < SLOTS) {
            room->booked[offset][slot] = 1;  // Mark as booked
            strncpy(room->user[offset][slot], username, MAX_NAME);  // Store username
        }
    }
    fclose(file);
}

// Finds the best available classroom matching requirements
int find_room(int required_size, int date_offset, int slot_index, char* best_room) {
    FILE* file = fopen("classroominfo.txt", "r");
    if (!file) {
        fprintf(stderr, "Error: Couldn't open classroom database\n");
        return 0;
    }

    Classroom rooms[MAX_ROOMS];
    int room_count = 0;
    int best_capacity = NO_ROOM_FOUND;  // Start with "no room found" state

    // Load all classrooms from master file
    while (room_count < MAX_ROOMS &&
        fscanf(file, "%s %d", rooms[room_count].id, &rooms[room_count].capacity) == 2) {

        // Initialize booking state
        memset(rooms[room_count].booked, 0, sizeof(rooms[room_count].booked));
        for (int d = 0; d < DAYS; d++) {
            for (int s = 0; s < SLOTS; s++) {
                rooms[room_count].user[d][s][0] = '\0';  // Clear username
            }
        }

        // Load existing bookings for this room
        load_classroom_status(&rooms[room_count]);
        room_count++;
    }
    fclose(file);

    // Search for best matching classroom
    for (int i = 0; i < room_count; i++) {
        Classroom* current = &rooms[i];

        // Skip rooms that are too small or already booked
        if (current->capacity < required_size) continue;
        if (current->booked[date_offset][slot_index] != 0) continue;

        // Found a candidate - check if it's better than current best
        if (best_capacity == NO_ROOM_FOUND || current->capacity < best_capacity) {
            best_capacity = current->capacity;
            strcpy(best_room, current->id);  // Update best match
        }
    }

    // Return whether we found a suitable room
    return best_capacity != NO_ROOM_FOUND;
}
