#include <stdio.h>
#include <time.h>
#include "header.h"

// Function 1: Access Date for menu
void get_date(int offset, char* date_str) {
    struct tm* tblock;
    time_t now;

    // Part1: obtain wanted date based on day_bias
    time(&now);                    // Get the current time
    now += offset * 24 * 60 * 60;        // To obtain the date dynamically
    tblock = localtime(&now);        // Convert to local time

    // Part2: Format the date into a string
    sprintf(date_str, "%02d/%02d/%04d",
        tblock->tm_mday,
        tblock->tm_mon + 1,      // tm_mon end at 11
        tblock->tm_year + 1900); // tm_year since 1900
}

// Function 2: Menu
int choose_date() {
    char today[20], tomorrow[20], after[20];
    int option;

    //Part1: Access the date
    get_date(0, today);           // Today's date
    get_date(1, tomorrow);        // Tomorrow's date
    get_date(2, after);           // Day after tomorrow's date

    //Part2: Display
    printf("\n--- Date Menu ---\n");
    printf("1. %s (Today)\n", today);
    printf("2. %s (Tomorrow)\n", tomorrow);
    printf("3. %s (Day after tomorrow)\n", after);
    printf("4. Exit\n");
    printf("Select date (1-4): ");
    scanf("%d", &option);

    //Partt3: Handle the option
    if (option == 4) {
        printf("Exiting...\nHave a nice day! :)\n");
        return -1;
    }
    else if (option >= 1 && option <= 3) {
        printf("You have selected option %d.\n", option);
        printf("The date is: %s\n", (option == 1) ? today : (option == 2) ? tomorrow : after);
        return option - 1;
    }
    else {
        printf("Invalid option. Please try again.\n");
        return choose_date(); // Return to the menu
    }
}
