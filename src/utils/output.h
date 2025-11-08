#pragma once

#include <cstdio>

// Определения цветов и стилей
#define RESET "\033[0m"
#define BOLD "\033[1m"
#define ITALIC "\033[3m"
#define UNDERLINE "\033[4m"

// Цвета текста
#define BLACK "\033[30m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define WHITE "\033[37m"

// Фоновые цвета
#define BG_BLACK "\033[40m"
#define BG_RED "\033[41m"
#define BG_GREEN "\033[42m"
#define BG_YELLOW "\033[43m"
#define BG_BLUE "\033[44m"
#define BG_MAGENTA "\033[45m"
#define BG_CYAN "\033[46m"
#define BG_WHITE "\033[47m"

#define printff(format_codes, ...)                                             \
  do {                                                                         \
    printf("%s", format_codes);                                                \
    printf(__VA_ARGS__);                                                       \
    printf("\033[0m\n");                                                       \
  } while (0)

#ifdef DEBUG_MODE
#define debug(fmt, ...)                                                        \
  do {                                                                         \
    printf(fmt, ##__VA_ARGS__);                                                \
    printf("\n");                                                              \
  } while (0)
#define debugi(fmt, ...)                                                       \
  do {                                                                         \
    printf(fmt, ##__VA_ARGS__);                                                \
  } while (0)
#define debugf(format_codes, fmt, ...)                                         \
  do {                                                                         \
    printf("%s", format_codes);                                                \
    printf("[%s:%d] ", __FILE__, __LINE__);                                    \
    printf(fmt, ##__VA_ARGS__);                                                \
    printf("\n");                                                              \
    printf("\033[0m\n");                                                       \
  } while (0)
#define loge(fmt, ...) logff(RED UNDERLINE, fmt, ##__VA_ARGS__)
#else
#define debug(fmt, ...)
#define debugi(fmt, ...)
#define debugf(format_codes, fmt, ...)
#endif
