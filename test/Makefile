###########################################################
################### Makefile Template #####################
###########################################################

# Compiler settings - Can be customized.
CXX = g++
CXXFLAGS = -c -Wall -Wextra -O2 -std=gnu++2a

# Files  Object Headers end Executable
HDRS=*.h
OBJS=main.o AI.o Net.o Layer.o Weights.o Gradient.o
BIN=output

################# Unless you now what you are doing do not modify this part ##################

all: $(BIN)

$(BIN): $(OBJS) $(HDRS)
	$(CXX) $(OBJS) -o $(BIN) -pthread
	@echo "Linked everything together"

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

AI.o: AI.cpp
	$(CXX) $(CXXFLAGS) -c AI.cpp

# Clean everything but doesn't delete the Saved_Weights file
.PHONY: clean
clean:
	$(RM) $(OBJS) $(BIN)

################## main compilation will work with implicit dependences ##################
