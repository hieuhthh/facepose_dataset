import os, shutil

def make_archive(source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)

from_path = '/home/lap14880/hieunmt/facepose/facepose_gendata/dataset'
to_path = '/home/lap14880/hieunmt/facepose/facepose_gendata/dataset.zip'
make_archive(from_path, to_path)