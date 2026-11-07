#include <stdio.h>
#include <string.h>
#include "header.h"

// Function: Writes the booking record to the corresponding classroom's .txt file
int save_booking(char* room, int day, int slot, char* user) {
    char filename[20];
    char date[20];

    // Generate the file name based on room ID
    sprintf(filename, "%s.txt", room);  

    // Get the formatted date string for the selected day
    get_date(day, date);                

    // Open the file in append mode (creates if not exists)
    FILE* fp = fopen(filename, "a");    
    if (!fp) {
        printf("Failed to open %s for writing.\n", filename);
        return 0;
    }


    fprintf(fp, "%s %d %s\n", date, slot, user);

    fclose(fp);
    return 1;
}
