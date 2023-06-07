CXX = g++
CXXFLAGS = -O2 $(shell root-config --cflags) -DNDEBUG

TARGET = run_mcmc
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)
LDFLAGS = -larmadillo $(shell root-config --glibs)
OBJECTS = $(SOURCES:.cpp=.o)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

profile: CXXFLAGS += -pg -fno-inline-functions -fno-inline-functions-called-once -fno-optimize-sibling-calls
profile: $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

