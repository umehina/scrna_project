package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

type testSampleBlock struct {
	name     string
	input    []string
	expected []string
}

// parseTestFile parses a test file into testSampleBlocks.
// It takes in a file path and returns a slice of testSampleBlock structs, each containing input and expected slices of strings.
func parseTestFile(path string) ([]testSampleBlock, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(b), "\n")
	var blocks []testSampleBlock
	var cur testSampleBlock
	mode := "none"
	// parse line by line
	for _, raw := range lines {
		ln := strings.TrimSpace(raw)
		if ln == "" {
			continue
		}
		if strings.HasPrefix(ln, "# Sample Input") {
			if len(cur.input)+len(cur.expected) > 0 {
				blocks = append(blocks, cur)
				cur = testSampleBlock{}
			}
			mode = "input"
			continue
		}
		if strings.HasPrefix(ln, "@ Sample Output") || strings.HasPrefix(ln, "@Sample Output") {
			mode = "output"
			continue
		}
		if strings.HasPrefix(ln, "#") || strings.HasPrefix(ln, "//") {
			continue
		}
		switch mode {
		case "input":
			cur.input = append(cur.input, ln)
		case "output":
			cur.expected = append(cur.expected, ln)
		}
	}
	// append last block
	if len(cur.input)+len(cur.expected) > 0 {
		blocks = append(blocks, cur)
	}
	return blocks, nil
}

// parseFloatLine parses a line into a slice of floats.
// It takes in a line string and the expected number of floats want. It returns a slice of floats and an error if any.
func parseFloatLine(ln string, want int) ([]float64, error) {
	fields := strings.Fields(ln)
	// check length
	if len(fields) < want {
		return nil, fmt.Errorf("expected %d numbers, got %d", want, len(fields))
	}
	vals := make([]float64, want)
	for i := 0; i < want; i++ {
		v, err := strconv.ParseFloat(fields[i], 64)
		if err != nil {
			return nil, err
		}
		vals[i] = v
	}
	return vals, nil
}

// parseBoolLine parses a line into a slice of bools.
// It takes in a line string and the expected number of bools want. It returns a slice of bools and an error if any.
func parseBoolLine(ln string, want int) ([]bool, error) {
	fields := strings.Fields(ln)
	// check length
	if len(fields) < want {
		return nil, fmt.Errorf("expected %d booleans, got %d", want, len(fields))
	}
	vals := make([]bool, want)
	for i := 0; i < want; i++ {
		v, err := strconv.ParseBool(fields[i])
		if err != nil {
			return nil, err
		}
		vals[i] = v
	}
	return vals, nil
}

// runFileTests runs tests defined in a file.
// It takes in a testing.T, test name, tolerance, and a function fn to run for each testSampleBlock.
func runFileTests(t *testing.T, name string, tol float64, fn func(b testSampleBlock, idx int, tol float64)) {
	path := filepath.Join(".", "testdata", "helpers", name+".txt")
	blocks, err := parseTestFile(path)
	if err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	for i, b := range blocks {
		fn(b, i, tol) // <-- pass tolerance down
	}
	fmt.Printf("Testing %s... done.\n", name)
}

// TestFloatEqualsFromFile takes in test cases from a file and tests the floatEquals function.
func TestFloatEqualsFromFile(t *testing.T) {
	runFileTests(t, "floatEquals", 0, func(sb testSampleBlock, idx int, _ float64) {
		if len(sb.input) == 0 || len(sb.expected) == 0 {
			t.Fatalf("block %d (%s) missing input or expected", idx, sb.name)
		}
		in, err := parseFloatLine(sb.input[0], 3)
		if err != nil {
			t.Fatalf("parse input %d (%s): %v", idx, sb.name, err)
		}
		exp, err := parseBoolLine(sb.expected[0], 1)
		if err != nil {
			t.Fatalf("parse expected %d (%s): %v", idx, sb.name, err)
		}
		got := floatEquals(in[0], in[1], in[2])
		want := exp[0]
		if got != want {
			t.Fatalf("case %d (%s): got %v want %v", idx, sb.name, got, want)
		}
	})
}
