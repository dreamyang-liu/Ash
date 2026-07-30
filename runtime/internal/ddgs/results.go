package ddgs

// Result is a single search result. It mirrors the dicts returned by the
// Python ddgs library: keys and value shapes are identical per category.
//
//	text:   title, href, body
//	images: title, image, thumbnail, url, height, width, source
//	news:   date, title, body, url, image, source
//	videos: title, content, description, duration, embed_html, embed_url,
//	        image_token, images, provider, published, publisher, statistics, uploader
//	books:  title, author, publisher, info, url, thumbnail
type Result map[string]any

// resultFields lists the dict keys per category, in the same order as the
// Python dataclass fields.
var resultFields = map[string][]string{
	"text":   {"title", "href", "body"},
	"images": {"title", "image", "thumbnail", "url", "height", "width", "source"},
	"news":   {"date", "title", "body", "url", "image", "source"},
	"videos": {
		"title", "content", "description", "duration", "embed_html", "embed_url",
		"image_token", "images", "provider", "published", "publisher", "statistics", "uploader",
	},
	"books": {"title", "author", "publisher", "info", "url", "thumbnail"},
}

// fieldNormalizers maps field name -> normalizer, mirroring
// BaseResult._normalizers in the Python implementation.
var fieldNormalizers = map[string]func(string) string{
	"title":     normalizeText,
	"body":      normalizeText,
	"href":      normalizeURL,
	"url":       normalizeURL,
	"thumbnail": normalizeURL,
	"image":     normalizeURL,
	"author":    normalizeText,
	"publisher": normalizeText,
	"info":      normalizeText,
}

// newResult creates a Result pre-populated with the empty fields of the
// given category, matching Python dataclass defaults.
func newResult(category string) Result {
	fields, ok := resultFields[category]
	if !ok {
		return Result{}
	}
	r := make(Result, len(fields))
	for _, f := range fields {
		switch f {
		case "images", "statistics": // videos category dict fields
			r[f] = map[string]any{}
		default:
			r[f] = ""
		}
	}
	return r
}

// Set assigns a value to a field, applying the same normalization the
// Python BaseResult.__setattr__ performs. A nil value is stored as nil
// (Python stores None), overriding the "" default.
func (r Result) Set(field string, value any) {
	if value == nil {
		r[field] = nil
		return
	}
	if field == "date" {
		r[field] = normalizeDate(value)
		return
	}
	if s, ok := value.(string); ok && s != "" {
		if normalizer, ok := fieldNormalizers[field]; ok {
			r[field] = normalizer(s)
			return
		}
	}
	r[field] = value
}

// getString returns the string value of a field, or "" if absent/not a string.
func (r Result) getString(field string) string {
	if v, ok := r[field]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// resultsAggregator deduplicates incoming results by the first matching
// cache field and counts occurrences. extract returns items sorted by
// descending frequency (insertion order breaks ties), mirroring
// ddgs.results.ResultsAggregator.
type resultsAggregator struct {
	cacheFields []string // checked in result-field order
	counter     map[string]int
	order       []string // insertion order of keys
	cache       map[string]Result
}

// aggregatorCacheFields mirrors {"href", "image", "url", "embed_url"}.
var aggregatorCacheFields = map[string]bool{
	"href":      true,
	"image":     true,
	"url":       true,
	"embed_url": true,
}

func newResultsAggregator(category string) *resultsAggregator {
	var fields []string
	for _, f := range resultFields[category] {
		if aggregatorCacheFields[f] {
			fields = append(fields, f)
		}
	}
	return &resultsAggregator{
		cacheFields: fields,
		counter:     map[string]int{},
		cache:       map[string]Result{},
	}
}

func (a *resultsAggregator) key(item Result) (string, bool) {
	for _, f := range a.cacheFields {
		if v, ok := item[f]; ok {
			if s, ok := v.(string); ok {
				return s, true
			}
			return "", false
		}
	}
	return "", false
}

func (a *resultsAggregator) len() int { return len(a.cache) }

// append registers an occurrence of item. The stored copy is replaced when
// the incoming item has a longer body (richer snippet), as in Python.
func (a *resultsAggregator) append(item Result) {
	key, ok := a.key(item)
	if !ok {
		return
	}
	existing, seen := a.cache[key]
	if !seen {
		a.cache[key] = item
		a.order = append(a.order, key)
	} else if len(item.getString("body")) > len(existing.getString("body")) {
		a.cache[key] = item
	}
	a.counter[key]++
}

func (a *resultsAggregator) extend(items []Result) {
	for _, item := range items {
		a.append(item)
	}
}

// extract returns results sorted by descending frequency; ties keep
// insertion order (same as Counter.most_common with Python dict ordering).
func (a *resultsAggregator) extract() []Result {
	keys := make([]string, len(a.order))
	copy(keys, a.order)
	stableSortByCountDesc(keys, a.counter)
	out := make([]Result, 0, len(keys))
	for _, k := range keys {
		out = append(out, a.cache[k])
	}
	return out
}

func stableSortByCountDesc(keys []string, counter map[string]int) {
	// insertion sort keeps stability; n is small (search results)
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && counter[keys[j]] > counter[keys[j-1]]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
}
