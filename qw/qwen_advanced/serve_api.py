"""Optional local FastAPI endpoint loading Qwen and the index once per process."""
import argparse
from threading import Lock

from ewoc_common import load_labels
from ewoc_classifier import QwenClassifier,add_inference_args,make_retriever


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_inference_args(p)
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=8000)
    args = p.parse_args()
    if args.retrieval_device != "cpu":
        raise ValueError("API retrieval uses CPU; reserve the GPU for vLLM. Use --retrieval-device cpu.")
    from fastapi import FastAPI
    from pydantic import BaseModel,Field
    import uvicorn
    labels = load_labels(args.labels)
    retriever = make_retriever(args,labels)
    classifier = QwenClassifier(args,labels)
    lock = Lock()
    app = FastAPI(title="EWOC Qwen Classifier",version="1.0")
    class Request(BaseModel):
        description: str = Field(min_length=1,max_length=50000)
        include_evidence: bool = False
    @app.get("/health")
    def health():
        return {"ready":True,"rag":retriever is not None,"adapter":bool(args.adapter),"classes":len(labels)}
    @app.post("/classify")
    def classify(request:Request):
        if not request.description.strip():
            from fastapi import HTTPException
            raise HTTPException(status_code=422,detail="Description cannot be blank")
        # Serialize synchronous vLLM plus retrieval. Run one worker per GPU.
        with lock:
            examples = retriever.retrieve(request.description,count=args.examples,
                                           max_per_type=args.max_per_type,pool=args.pool,
                                           rerank_top=args.rerank_top) if retriever else []
            result = classifier.classify_batch([request.description],[examples])[0]
        if not request.include_evidence:
            result.pop("evidence",None)
        return result
    # The object form prevents accidentally loading multiple GPU copies with workers/reload.
    uvicorn.run(app,host=args.host,port=args.port,workers=1)


if __name__ == "__main__":
    main()
