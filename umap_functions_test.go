package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"gonum.org/v1/gonum/mat"
)

// ---------- generic parser for # / @ format ----------

type ioCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

// parseIOCases reads a txt file in your custom format:
//
// # some description for input
// <input line>
// @ some description for output
// <output line>
// [blank line]
// ... repeats ...
func parseIOCases(path string) ([]ioCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var cases []ioCase
	scanner := bufio.NewScanner(f)

	var (
		currentName      string
		waitingForInput  bool
		waitingForOutput bool
		pendingInput     string
		lineNum          int
	)

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// description for next input
			currentName = strings.TrimSpace(line[1:])
			waitingForInput = true
			waitingForOutput = false
			pendingInput = ""

		case strings.HasPrefix(line, "@"):
			// description for next output
			if currentName == "" {
				currentName = strings.TrimSpace(line[1:])
			}
			waitingForOutput = true
			waitingForInput = false

		default:
			if waitingForInput {
				// this is the input line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output line
				cases = append(cases, ioCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				// reset
				currentName = ""
				waitingForOutput = false
				pendingInput = ""
			} else {
				return nil, fmt.Errorf("parse error at line %d: unexpected data line without #/@", lineNum)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

// ---------- float helpers ----------

func floatAlmostEqual(a, b, tol float64) bool {
	if tol < 0 {
		return false
	}
	return math.Abs(a-b) <= tol
}

func sliceAlmostEqual(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !floatAlmostEqual(a[i], b[i], tol) {
			return false
		}
	}
	return true
}

func matrixAlmostEqual(a, b [][]float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if !floatAlmostEqual(a[i][j], b[i][j], tol) {
				return false
			}
		}
	}
	return true
}

func tcName(idx int, name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return "case_" + strconv.Itoa(idx)
	}
	name = strings.ReplaceAll(name, " ", "_")
	return strconv.Itoa(idx) + "_" + name
}

// helper to unmarshal JSON from a single line and give nice errors in tests
func mustUnmarshalJSON[T any](t *testing.T, s string) T {
	var out T
	if err := json.Unmarshal([]byte(s), &out); err != nil {
		t.Fatalf("failed to unmarshal JSON %q: %v", s, err)
	}
	return out
}

// ========== Test ComputeDistanceMatrix ===============
func TestComputeDistanceMatrix_FromFile(t *testing.T) {
	const testFile = "UMAP_tests/computeDistanceMatrix_tests.txt"

	cases, err := parseIOCases(testFile)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no test cases in %s", testFile)
	}

	for idx, tc := range cases {
		t.Run(tcName(idx, tc.Name), func(t *testing.T) {
			// Input: JSON [][]float64 of points
			points := mustUnmarshalJSON[[][]float64](t, tc.InputLine)

			// Call your function (assumed signature)
			distMtx := computeDistanceMatrix(points, Euclidean)

			// Convert mat.Dense to [][]float64 for comparison
			r, c := distMtx.Dims()
			got := make([][]float64, r)
			for i := 0; i < r; i++ {
				got[i] = make([]float64, c)
				for j := 0; j < c; j++ {
					got[i][j] = distMtx.At(i, j)
				}
			}

			// Expected: JSON [][]float64
			want := mustUnmarshalJSON[[][]float64](t, tc.OutputLine)

			if !matrixAlmostEqual(got, want, 1e-6) {
				t.Fatalf("distance matrix mismatch.\n got:  %v\n want: %v", got, want)
			}
		})
	}
}

// ========== Test BuildKNNForUMAP =================
type knnInput struct {
	K    int         `json:"k"`
	Dist [][]float64 `json:"dist"`
}

type knnOutput struct {
	Idx  [][]int     `json:"idx"`
	Dist [][]float64 `json:"dist"`
}

func TestBuildKNNForUMAP_FromFile(t *testing.T) {
	const testFile = "UMAP_tests/BuildKNNForUMAP_tests.txt"

	cases, err := parseIOCases(testFile)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no test cases in %s", testFile)
	}

	for idx, tc := range cases {
		t.Run(tcName(idx, tc.Name), func(t *testing.T) {
			// Input
			var in knnInput
			if err := json.Unmarshal([]byte(tc.InputLine), &in); err != nil {
				t.Fatalf("failed to unmarshal input: %v", err)
			}

			r := len(in.Dist)
			c := 0
			if r > 0 {
				c = len(in.Dist[0])
			}
			dense := mat.NewDense(r, c, nil)
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					dense.Set(i, j, in.Dist[i][j])
				}
			}

			knnIdx, knnDist := BuildKNNForUMAP(dense, in.K)

			// Expected
			var out knnOutput
			if err := json.Unmarshal([]byte(tc.OutputLine), &out); err != nil {
				t.Fatalf("failed to unmarshal output: %v", err)
			}

			// Compare idx
			if len(knnIdx) != len(out.Idx) {
				t.Fatalf("idx length mismatch: got %d, want %d", len(knnIdx), len(out.Idx))
			}
			for i := range knnIdx {
				if len(knnIdx[i]) != len(out.Idx[i]) {
					t.Fatalf("idx[%d] length mismatch: got %d, want %d",
						i, len(knnIdx[i]), len(out.Idx[i]))
				}
				for j := range knnIdx[i] {
					if knnIdx[i][j] != out.Idx[i][j] {
						t.Fatalf("idx[%d][%d]=%d, want %d", i, j, knnIdx[i][j], out.Idx[i][j])
					}
				}
			}

			// Compare dist
			if !matrixAlmostEqual(knnDist, out.Dist, 1e-6) {
				t.Fatalf("dist mismatch.\n got:  %v\n want: %v", knnDist, out.Dist)
			}
		})
	}
}

// =========== Test ComputeRhoSigma =============
type rhoSigmaInput struct {
	K       int         `json:"k"`
	KnnDist [][]float64 `json:"knnDist"`
}

type rhoSigmaOutput struct {
	Rho []float64 `json:"rho"`
	// we ignore sigma in this test
}

func TestComputeRhoSigma_FromFile(t *testing.T) {
	const testFile = "UMAP_tests/computeRhoSigma_tests.txt"

	cases, err := parseIOCases(testFile)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no test cases in %s", testFile)
	}

	for idx, tc := range cases {
		t.Run(tcName(idx, tc.Name), func(t *testing.T) {
			var in rhoSigmaInput
			if err := json.Unmarshal([]byte(tc.InputLine), &in); err != nil {
				t.Fatalf("unmarshal input: %v", err)
			}
			var out rhoSigmaOutput
			if err := json.Unmarshal([]byte(tc.OutputLine), &out); err != nil {
				t.Fatalf("unmarshal output: %v", err)
			}

			rho, _ := computeRhoSigma(in.KnnDist, in.K)

			if !sliceAlmostEqual(rho, out.Rho, 1e-6) {
				t.Fatalf("rho mismatch.\n got:  %v\n want: %v", rho, out.Rho)
			}
		})
	}
}

// =========== Test buildDirectedProbs ==============
type directedInput struct {
	KnnIdx  [][]int       `json:"knnIdx"`
	KnnDist [][]float64   `json:"knnDist"`
	Rho     []float64     `json:"rho"`
	Sigma   []float64     `json:"sigma"`
}

type triple struct {
	I int
	J int
	P float64
}

func TestBuildDirectedProbs_FromFile(t *testing.T) {
	const testFile = "UMAP_tests/buildDirectedProbs_tests.txt"

	cases, err := parseIOCases(testFile)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no test cases in %s", testFile)
	}

	for idx, tc := range cases {
		t.Run(tcName(idx, tc.Name), func(t *testing.T) {
			var in directedInput
			if err := json.Unmarshal([]byte(tc.InputLine), &in); err != nil {
				t.Fatalf("unmarshal input: %v", err)
			}

			// expected triples: [[i,j,p], ...]
			var rawTriples [][]float64
			if err := json.Unmarshal([]byte(tc.OutputLine), &rawTriples); err != nil {
				t.Fatalf("unmarshal output: %v", err)
			}
			exp := make(map[pair]float64)
			for _, tt := range rawTriples {
				if len(tt) != 3 {
					t.Fatalf("expected triple [i,j,p], got %v", tt)
				}
				i := int(tt[0])
				j := int(tt[1])
				p := tt[2]
				exp[pair{I: i, J: j}] = p
			}

			got := buildDirectedProbs(in.KnnIdx, in.KnnDist, in.Rho, in.Sigma)

			if len(got) != len(exp) {
				t.Fatalf("len(got)=%d, len(exp)=%d", len(got), len(exp))
			}
			for key, wantP := range exp {
				gotP, ok := got[key]
				if !ok {
					t.Fatalf("missing key %+v in got", key)
				}
				if !floatAlmostEqual(gotP, wantP, 1e-6) {
					t.Fatalf("p[%v] = %v, want %v", key, gotP, wantP)
				}
			}
		})
	}
}

// =========== Test buildFuzzyGraph =============
type fuzzyInput struct {
	K       int         `json:"k"`
	KnnIdx  [][]int     `json:"knnIdx"`
	KnnDist [][]float64 `json:"knnDist"`
}

func TestBuildFuzzyGraph_FromFile(t *testing.T) {
	const testFile = "UMAP_tests/buildFuzzyGraph_tests.txt"

	cases, err := parseIOCases(testFile)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no test cases in %s", testFile)
	}

	for idx, tc := range cases {
		t.Run(tcName(idx, tc.Name), func(t *testing.T) {
			var in fuzzyInput
			if err := json.Unmarshal([]byte(tc.InputLine), &in); err != nil {
				t.Fatalf("unmarshal input: %v", err)
			}

			// Run buildFuzzyGraph (it calls computeRhoSigma + buildDirectedProbs internally)
			edges := buildFuzzyGraph(in.KnnIdx, in.KnnDist, in.K)

			// Expected: list of [i,j,weight]
			var rawTriples [][]float64
			if err := json.Unmarshal([]byte(tc.OutputLine), &rawTriples); err != nil {
				t.Fatalf("unmarshal output: %v", err)
			}
			exp := make(map[[2]int]float64)
			for _, tt := range rawTriples {
				if len(tt) != 3 {
					t.Fatalf("expected triple [i,j,w], got %v", tt)
				}
				i := int(tt[0])
				j := int(tt[1])
				w := tt[2]
				if i > j {
					i, j = j, i
				}
				exp[[2]int{i, j}] = w
			}

			got := make(map[[2]int]float64)
			for _, e := range edges {
				i, j := e.I, e.J
				if i > j {
					i, j = j, i
				}
				got[[2]int{i, j}] = e.Weight
			}

			if len(got) != len(exp) {
				t.Fatalf("len(got)=%d, len(exp)=%d", len(got), len(exp))
			}
			for key, want := range exp {
				g, ok := got[key]
				if !ok {
					t.Fatalf("missing edge %v in got", key)
				}
				if !floatAlmostEqual(g, want, 1e-6) {
					t.Fatalf("edge %v weight = %v, want %v", key, g, want)
				}
			}
		})
	}
}