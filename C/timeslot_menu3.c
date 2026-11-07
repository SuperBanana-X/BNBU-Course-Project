#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include "header.h"

// the function is used to display the available time slots of that day or future, and return the choise of users, -1 is sign for exit
int choose_slot(int date_offset) {
    // difine a structure to store all the context
    const char* slots[] = {
        "9:00 - 9:50", "10:00 - 10:50", "11:00 - 11:50",
        "12:00 - 12:50", "13:00 - 13:50", "14:00 - 14:50",
        "15:00 - 15:50", "16:00 - 16:50"
    };

    // get the current time,hours and minutes
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    int currH = t->tm_hour, currM = t->tm_min;

    bool valid[8] = { false };   // mark wether it is available
    int index = 1;       // the index of menu, starting from 1

    printf("\n--- Available Timeslots ---\n");


    for (int i = 0; i < 8; i++) {
        int startH = 9 + i;  // every day starts from 9 o'clock
        // if it is the future, then every slot can be booked
        // if it is today,then we need to consider whether it's expired
        if (date_offset != 0 || currH < startH || (currH == startH && currM < 0)) {
            valid[i] = true;
            printf("%d. %s\n", index++, slots[i]);
        }
    }

    printf("%d. Exit\n", index);  // the last one is exit

    int choice;
    printf("Choose slot: ");
    scanf("%d", &choice);

    if (choice == index) return -1;  // user choose to exit

    // give index to available choises
    int count = 0;
    for (int i = 0; i < 8; i++) {
        if (valid[i]) {
            count++;
            if (count == choice) return i;
        }
    }

    return -1;  // for the time doesn't match or input invalid
}
