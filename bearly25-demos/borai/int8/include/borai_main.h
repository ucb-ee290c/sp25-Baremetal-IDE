#ifndef BORAI_MAIN_WRAPPER_H
#define BORAI_MAIN_WRAPPER_H

/* Uniquely-named wrapper for borai's int8 main.h. Used only by the combined KWS+Llama demo
 * (c2c-demos/bearly-kws-llama), where the KWS consumer TU also ships a header named "main.h".
 * A quoted include is searched from THIS file's directory first, so it resolves to the sibling
 * int8/include/main.h regardless of the target's -I ordering. */
#include "main.h"

#endif /* BORAI_MAIN_WRAPPER_H */
