CXX=g++
CXXFLAGS=-std=c++17 -Wall -Wextra -Wpedantic

SRC_DIR=micrograd
TEST_DIR=test
BUILD_DIR=build

SRC=$(wildcard $(SRC_DIR)/*.cpp)
TEST_SRC=$(wildcard $(TEST_DIR)/*.cpp)

OBJ=$(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC))
TEST_OBJ=$(patsubst $(TEST_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(TEST_SRC))

TARGET=$(BUILD_DIR)/test

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Linked $(BIN)"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
	@echo "Everything clean"
