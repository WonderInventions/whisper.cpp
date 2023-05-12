package whisper_test

import (
	"os"
	"sync"
	"testing"

	// Packages
	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/go-audio/wav"
	assert "github.com/stretchr/testify/assert"
)

const (
	ModelPath  = "../../models/ggml-tiny.bin"
	SamplePath = "../../samples/jfk.wav"
)

func Test_Whisper_000(t *testing.T) {
	assert := assert.New(t)
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	// Load model
	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	assert.NoError(model.Close())

	t.Log("languages=", model.Languages())
}

func Test_Whisper_001(t *testing.T) {
	assert := assert.New(t)
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	// Load model
	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

	// Get context for decoding
	ctx, err := model.NewContext()
	assert.NoError(err)
	assert.NotNil(ctx)

}

func Test_Whisper_002(t *testing.T) {
	assert := assert.New(t)
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	// Load sample
	samplefile, err := os.Open(SamplePath)
	assert.NoError(err)
	d := wav.NewDecoder(samplefile)
	buf, err := d.FullPCMBuffer()
	assert.NoError(err)
	data := buf.AsFloat32Buffer().Data

	// Load model
	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

	// Get 2 contexts for decoding
	ctx, err := model.NewContext()
	assert.NoError(err)
	assert.NotNil(ctx)
	ctx2, err := model.NewContext()
	assert.NoError(err)
	assert.NotNil(ctx2)

	// Decode concurrently.
	var wg sync.WaitGroup
	wg.Add(2)
	go processSample(t, ctx, data, &wg)
	go processSample(t, ctx2, data, &wg)
	wg.Wait()
}

func processSample(t *testing.T, ctx whisper.Context, data []float32, wg *sync.WaitGroup) {
	err := ctx.Process(data, func(seg whisper.Segment) {
		t.Logf("text: %v", seg.Text)
	})
	assert.NoError(t, err)
	wg.Done()
}
