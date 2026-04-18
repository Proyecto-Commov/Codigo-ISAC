# main_pipeline.py
from Hugo.prototype.config import PipelineConfig
from Hugo.prototype.raw_data_loader import RawDataLoader
from Hugo.prototype.format_adapter import FormatAdapter
from Hugo.prototype.ofdm_frame_parser import OFDMFrameParser
from Hugo.prototype.pilot_extractor import PilotExtractor
from Hugo.prototype.channel_estimator import ChannelEstimator
from Hugo.prototype.phase_doppler_estimator import PhaseDopplerEstimator
from Hugo.prototype.clutter_filter import ClutterFilter
from Hugo.prototype.slowtime_builder import SlowTimeBuilder
from Hugo.prototype.spectrogram_generator import SpectrogramGenerator


class MicroDopplerPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.loader = RawDataLoader()
        self.adapter = FormatAdapter()
        self.parser = OFDMFrameParser(cfg)
        self.pilot_extractor = PilotExtractor(cfg)
        self.channel_estimator = ChannelEstimator()
        self.phase_estimator = PhaseDopplerEstimator()
        self.clutter_filter = ClutterFilter(cfg)
        self.slow_builder = SlowTimeBuilder()
        self.spec_generator = SpectrogramGenerator(cfg)

    def run_from_file(self, filepath: str, source_type: str = "generic", **kwargs):
        block = self.loader.load(filepath, source_type=source_type, **kwargs)
        return self.run(block)

    def run(self, block):
        block = self.adapter.adapt(block)
        grid = self.parser.parse(block)
        pilots = self.pilot_extractor.extract(grid)
        channel = self.channel_estimator.estimate(pilots)
        slow = self.phase_estimator.estimate(channel)
        slow = self.clutter_filter.apply(slow)
        slow = self.slow_builder.build(slow, mode="per_pilot")
        spec = self.spec_generator.generate(slow)

        return {
            "grid": grid,
            "pilots": pilots,
            "channel": channel,
            "slowtime": slow,
            "spectrogram": spec,
        }


if __name__ == "__main__":
    cfg = PipelineConfig()
    pipe = MicroDopplerPipeline(cfg)
    print("Pipeline instanciado correctamente.")