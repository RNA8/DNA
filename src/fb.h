#pragma once
/*
 * Minimal read-only FlatBuffers deserializer.
 *
 * FlatBuffers binary layout (all values little-endian):
 *   File     : u32 root_offset, [u8 file_id[4]], data...
 *   Table    : i32 soffset_to_vtable, then fields at offsets stored in vtable
 *   Vtable   : u16 vtable_size, u16 object_size, u16 field_offset[N]
 *              field_offset[i] == 0 means field i is absent (use default)
 *   Vector   : u32 element_count, element[0..count-1]
 *   String   : u32 byte_length, char[length], '\0'
 *   Nested   : stored as u32 offset from the field location
 *   Scalars  : stored inline at the field offset inside the object
 */

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

static inline uint16_t fb_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static inline uint32_t fb_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline int32_t fb_i32(const uint8_t *p) { return (int32_t)fb_u32(p); }

/* Root table pointer from the start of the file buffer. */
static inline const uint8_t *fb_root(const uint8_t *buf) {
    return buf + fb_u32(buf);
}

/*
 * Return a pointer to field field_id inside a table, or NULL if absent.
 * field_id is 0-based (matches declaration order in the .fbs schema).
 */
static inline const uint8_t *fb_field(const uint8_t *table, int field_id) {
    int32_t      soff  = fb_i32(table);
    const uint8_t *vt  = table - soff;
    uint16_t     vsz   = fb_u16(vt);
    uint16_t     fpos  = (uint16_t)(4 + field_id * 2);
    if (fpos >= vsz) return NULL;
    uint16_t     foff  = fb_u16(vt + fpos);
    return (foff == 0) ? NULL : table + foff;
}

/* Dereference an offset field to reach a nested table. */
static inline const uint8_t *fb_table(const uint8_t *fp) {
    return fp + fb_u32(fp);
}

/* Dereference an offset field to reach a vector.
 * Returns pointer to element 0; writes element count if count != NULL. */
static inline const uint8_t *fb_vec(const uint8_t *fp, uint32_t *count) {
    const uint8_t *v = fp + fb_u32(fp);
    if (count) *count = fb_u32(v);
    return v + 4;
}

/* Dereference an offset field to reach a string (past the u32 length). */
static inline const char *fb_str(const uint8_t *fp) {
    const uint8_t *p = fp + fb_u32(fp);
    return (const char *)(p + 4);
}

/* Typed field readers with defaults. */
static inline int8_t   fb_i8  (const uint8_t *t, int f, int8_t   d) { const uint8_t *p = fb_field(t,f); return p ? *(int8_t*)p   : d; }
static inline uint8_t  fb_u8  (const uint8_t *t, int f, uint8_t  d) { const uint8_t *p = fb_field(t,f); return p ? *p              : d; }
static inline int32_t  fb_fi32(const uint8_t *t, int f, int32_t  d) { const uint8_t *p = fb_field(t,f); return p ? fb_i32(p)      : d; }
static inline uint32_t fb_fu32(const uint8_t *t, int f, uint32_t d) { const uint8_t *p = fb_field(t,f); return p ? fb_u32(p)      : d; }
static inline bool     fb_bool(const uint8_t *t, int f, bool     d) { const uint8_t *p = fb_field(t,f); return p ? (bool)(*p)     : d; }
static inline float    fb_f32 (const uint8_t *t, int f, float    d) {
    const uint8_t *p = fb_field(t,f);
    if (!p) return d;
    float v; memcpy(&v, p, 4); return v;
}
