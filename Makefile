# tiny_yolo: memory-optimized YOLOv8 inference engine (C++17 standard library only).
#
#   make                   build ./tiny_yolo (static on Linux, dynamic on macOS)
#   make dynamic           dynamically linked binary
#   make static            fully static binary (Linux; nothing else needed on the target)
#   make ARCH=-mcpu=cortex-a72                       tune for a specific CPU
#   make CXX=aarch64-linux-gnu-g++ ARCH=-mcpu=cortex-a53   cross-compile example
CXX ?= c++
ARCH ?= -march=native
CXXFLAGS ?= -O3 -std=c++17 $(ARCH) -ffp-contract=fast -fno-math-errno -Wall -Wextra -Wpedantic
LDFLAGS ?= -pthread

SOURCES = src/app.cpp src/graph.cpp src/layers.cpp src/conv.cpp src/conv_chain.cpp \
          src/memory_planner.cpp src/thread_pool.cpp src/params.cpp src/image.cpp
HEADERS = $(wildcard src/*.hpp)

ifeq ($(shell uname -s),Darwin)
DEFAULT_TARGET = dynamic
else
DEFAULT_TARGET = static
endif

.PHONY: all static dynamic clean
all: $(DEFAULT_TARGET)

static: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -ffunction-sections -fdata-sections -Isrc $(SOURCES) -o tiny_yolo \
	    -static $(LDFLAGS) -Wl,--gc-sections -s

dynamic: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -Isrc $(SOURCES) -o tiny_yolo $(LDFLAGS)

clean:
	rm -f tiny_yolo
