#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "header.h"

// Define a 'User' Structure
struct User {
    char name[20];
    int password;
};

// Login function: Verify username and password to three time at most
int login(char* user) {
    char input_name[20];
    int input_pwd;
    struct User u;

    // Save username and password of authorized users
    FILE* fp = fopen("Authorized_User.txt", "r");
    if (!fp) {
        printf("Failed to open Authorized_User.txt\n");
        return 0;
    }

    printf("=== Welcome to Classroom Booking ===\n");

    // Try at most three times
    for (int i = 0; i < 3; i++) {
        printf("Username: ");
        scanf("%s", input_name);
        printf("Password: ");
        scanf("%d", &input_pwd);

        rewind(fp);  // reset file pointer to the beginning

        // Check if the input matches any authorized user
        while (fscanf(fp, "%s %d", u.name, &u.password) != EOF) {
            if (strcmp(input_name, u.name) == 0 && input_pwd == u.password) {
                strcpy(user, input_name);  // store the username if it matches
                printf("Login successful.\n");
                fclose(fp);
                return 1;
            }
        }

        printf("Invalid name or password!\n");
    }

    printf("You have failed three times!\n");
    fclose(fp);
    return 0;
}
