# TODO - VGS Chatbot Improvements

## 🚨 Priority Issues

**Critical Issues:**
- [ ] **Package-wide logging implementation** - Add comprehensive logging/tracing across all services for debugging and monitoring
- [ ] **Document chunking improvements** - Ensure sections with "Notes" are not split - these contain critical caveats for tables/sections
- [ ] **Admin system health page** - Update health dashboard to show relevant metrics for current system architecture
- [ ] **Source citation accuracy** - "Sources" always shows exactly two references (DHO + GASO) instead of actual chunks used - verify chunk fetching

**Infrastructure Needs:**
- [ ] Implement structured logging with correlation IDs for request tracing
- [ ] Enhanced document structure preservation during chunking
- [ ] Real-time system health monitoring with actionable metrics
- [ ] Chunk-level source attribution validation

**Result:** Production-ready retrieval system with research-grade accuracy combining the best of semantic similarity and traditional search methods.

## 🔄 Future Feature Enhancements

### 5. Authentication and Database
**Status:** Future Enhancement
- [ ] Implement proper user registration and management via email validation
- [ ] Add role-based access control

### 6. User Experience Improvements
**Status:** Medium Priority
- [ ] Better error messages
- [ ] User feedback (thumbs up thumbs down)
- [ ] Document upload progress indicators
- [ ] Search result ranking and explanation
- [ ] Chat history persistence

## Development Notes

**Priority Order:**
1. ✅ **Document search and retrieval** (COMPLETED - enhanced chunking & layered responses)
2. ✅ **Vector database implementation** (COMPLETED - ChromaDB with hybrid search)
3. ✅ **RAG embedding optimization** (COMPLETED - state-of-the-art multi-stage pipeline)
4. ✅ **Advanced retrieval quality** (COMPLETED - BM25 + RRF + cross-encoder reranking)
5. [ ] Remove all previous document retrievers that are not MongoDB Vector Search
5. ✅ Migrate to MongoDB Atlas
6. ✅ Restrict user signup to only @rafac.mod.gov.uk or @mod.uk domains
7. [ ] Performance improvements
8. [ ] Remove sharepoint mentions
9. [ ] Implement structured logging, package-wide
10. [ ] Email validation
11. [ ] Remove mention of Docker
12. [ ] Remove any remaining test or debug files that are not in `tests/`
13. [ ] Remove ChromaDB mentions
14. [ ] Tidy up dependencies
15. [ ] System health admin page needs updated

**Testing Priority:**
- End-to-end testing of multi-stage retrieval pipeline
- Validate cross-encoder reranking accuracy improvements
- Test document-level summary embeddings for broad queries
- Performance testing with larger document sets and BM25 indexing
- Verify deduplication effectiveness and metadata quality
