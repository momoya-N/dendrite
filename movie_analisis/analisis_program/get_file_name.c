#include <stdio.h>
#include <dirent.h>
int main() {
    DIR *dir;
    struct dirent *entry;

    dir = opendir("/mnt/d/dendrite_data/edited_data/edited_movie");
    if (dir == NULL) {
        perror("ディレクトリを開けませんでした");
        return 1;
    }
    while ((entry = readdir(dir)) != NULL) {
        printf("%s\n", entry->d_name);
    }
    closedir(dir);
    return 0;
}