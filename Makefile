CXX      ?= clang++
CXXFLAGS ?= -O3 -std=c++17
TARGET    = hrp_bench

.PHONY: build run clean

build: $(TARGET)

$(TARGET): bench.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) bench.cpp

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)
