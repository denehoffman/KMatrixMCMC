CXX = g++
CXXFLAGS = -std=c++11 -O2 $(shell root-config --cflags)

TARGET = run_mcmc
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)
LDFLAGS = -larmadillo $(shell root-config --glibs)
OBJECTS = $(SOURCES:.cpp=.o)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

clean:
	rm -f $(OBJECTS) $(TARGET)

