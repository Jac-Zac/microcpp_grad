CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++17
SRC = engine.cpp main.cpp engine.hpp
OBJ = $(SRC:.cpp=.o)
TARGET = output

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
