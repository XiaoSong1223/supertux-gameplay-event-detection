"""Development-only diagnostic. Never used to choose evaluation annotations."""
import json,time
from pathlib import Path
import cv2
from PIL import Image, ImageDraw
from gameplay_events.detection import SpriteDetector

ROOT=Path(__file__).resolve().parents[1]
cv2.setNumThreads(1)
detector=SpriteDetector(ROOT/'assets/templates/detector.json')
tiles=[]; records=[]
for sec in (5,10,15,20,30,40,50,60,70,80,90):
    frame=cv2.imread(str(ROOT/f'outputs/dev_{sec:03}.png'))
    start=time.perf_counter(); detections=detector.detect(frame); elapsed=time.perf_counter()-start
    print(sec,len(detections),round(elapsed,3),[(d.label,tuple(map(int,d.bbox)),round(d.confidence,2),d.subtype) for d in detections],flush=True)
    records.append({'timestamp':sec,'seconds':elapsed,'detections':[d.__dict__ for d in detections]})
    for d in detections:
        x1,y1,x2,y2=map(int,d.bbox);color=(0,210,255) if d.label=='coin' else (60,70,245)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
        cv2.putText(frame,f'{d.subtype} {d.confidence:.2f}',(x1,max(35,y1-2)),0,.28,color,1,cv2.LINE_AA)
    cv2.imwrite(str(ROOT/f'outputs/dev_detected_{sec:03}.png'),frame)
    image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));image.thumbnail((427,240))
    tile=Image.new('RGB',(427,262),'white');tile.paste(image);ImageDraw.Draw(tile).text((5,244),f'{sec}s | {len(detections)} objects | {elapsed:.2f}s/frame',fill='black');tiles.append(tile)
sheet=Image.new('RGB',(427*3,262*4),'#ddd')
for i,im in enumerate(tiles):sheet.paste(im,((i%3)*427,(i//3)*262))
sheet.save(ROOT/'outputs/detection_contact_sheet.jpg')
(ROOT/'outputs/dev_detection_preview.json').write_text(json.dumps(records,indent=2)+'\n')
