"""The package consists of modules for DNA sequence tasks."""

from .mpra.mpra_datamodule import DNASeqMPRADataset, DNASeqMPRADataModule
from .splice_site.splice_site_datamodule import (
    DNASeqSpliceSiteDataModule,
    DNASeqSpliceSiteDataset,
)

from .covid.covid_datamodule import DNASeqCovidDataset, DNASeqCovidDataModule
from .drosophila_enhancer.drosophila_enhancer_datamodule import (
    DNASeqDrosophilaEnhancerDataset,
    DNASeqDrosophilaEnhancerDataModule,
)
from .epigenetic_marks.epigenetic_marks_datamodule import (
    DNASeqEpigeneticMarksDataModule,
    DNASeqEpigeneticMarksDataset,
)
from .promoter.promoter_datamodule import (
    DNASeqPromoterDataset,
    DNASeqPromoterDataModule,
)

from .core_promoter.core_promoter_datamodule import (
    DNASeqCorePromoterDataset,
    DNASeqCorePromoterDataModule,
)
from .transcription_factor.transcription_factor_datamodule import (
    DNASeqTranscriptionFactorDataModule,
    DNASeqTranscriptionFactorDataset,
)
from .chromatin_profile.chromatin_profile_datamodule import (
    StreamingDNASeqChromatinProfileDataset,
    StreamingDNASeqChromatinProfileDataModule,
    DNASeqChromatinProfileDataset,
    DNASeqChromatinProfileDataModule,
)

__all__ = [
    "DNASeqPromoterDataset",
    "DNASeqPromoterDataModule",
    "DNASeqCorePromoterDataset",
    "DNASeqCorePromoterDataModule",
    "DNASeqCorePromoterDataset",
    "DNASeqCorePromoterDataModule",
    "DNASeqMPRADataset",
    "DNASeqMPRADataModule",
    "DNASeqSpliceSiteDataModule",
    "DNASeqSpliceSiteDataset",
    "DNASeqCovidDataset",
    "DNASeqCovidDataModule",
    "DNASeqDrosophilaEnhancerDataset",
    "DNASeqDrosophilaEnhancerDataModule",
    "DNASeqEpigeneticMarksDataset",
    "DNASeqEpigeneticMarksDataModule",
    "DNASeqTranscriptionFactorDataModule",
    "DNASeqTranscriptionFactorDataset",
    "StreamingDNASeqChromatinProfileDataset",
    "StreamingDNASeqChromatinProfileDataModule",
    "DNASeqChromatinProfileDataset",
    "DNASeqChromatinProfileDataModule",
]
