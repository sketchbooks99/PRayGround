#pragma once 

#define PG_INTERSECTION(name) __intersection__pg_##name
#define PG_INTERSECTION_TEXT(name) "__intersection__pg_"#name

#include <prayground/shape/box.h>
#include <prayground/shape/cylinder.h>
#include <prayground/shape/plane.h>
#include <prayground/shape/sphere.h>

