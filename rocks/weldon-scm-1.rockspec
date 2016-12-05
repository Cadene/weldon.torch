package = "weldon"
version = "scm-1"

source = {
   url = "git://github.com/Cadene/weldon.torch",
   tag = "master"
}

description = {
   summary = "Weldon Pooling for Torch7 nn",
   detailed = [[
Torch7 Implementation of Weldon Pooling
   ]],
   homepage = "https://github.com/Cadene/weldon.torch",
   license = "MIT License"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0"
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      LUA_PATH="$(LUADIR)",
      LUA_CPATH="$(LIBDIR)"
   }
}