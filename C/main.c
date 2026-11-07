#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "header.h"

int main() {
    char user[20];  // Stores the username of the currently logged-in user

    // Step 1: Prompt user to log in
    // If login fails after 3 attempts, exit the program
    if (!login(user)) {
        return 0;
    }

    // Step 2: Enter the booking loop
    // The user can make multiple bookings in one session
    while (1) {
        // Ask the user to choose a booking date:
        // 0 = today, 1 = tomorrow, 2 = day after tomorrow
        int day = choose_date();
        if (day == -1) {
            break;  // User selected "Exit"
        }
        // Ask the user to choose an available timeslot on the selected day
        // For today, timeslots that have already passed will be hidden
        int slot = choose_slot(day);
        if (slot == -1) {     // User selected "Exit"
            break;
        }  

        // Prompt the user to enter the number of seats they need
        // match a classroom with capacity >= this number
        int size;
        printf("Input the size of classroom: ");
        scanf("%d", &size);

        // Attempt to find a suitable classroom that is available
        char room[10];
        if (!find_room(size, day, slot, room)) {
            // If no classroom is available, let the user know
            printf("No available room!\n");
        }
        else {
            // Found a matching classroom ¡ª show it and record the booking
            printf("Allocated room: %s\n", room);
            save_booking(room, day, slot, user);
        }

        // Ask the user whether they want to make another booking
        char cont;
        printf("Continue booking? (y/n): ");
        scanf(" %c", &cont); 
        if (cont != 'y' && cont != 'Y') break;  // Exit loop if user says no
    }

    
    printf("Thank you for using the booking system!\n");
    return 0;
}
